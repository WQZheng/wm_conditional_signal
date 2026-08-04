"""P2v2: Improved RSSM training with richer signal representation + better eval.

Key changes vs P2:
  - Signal: one-hot [G,Y,R] (3D) → distance-weighted [G,Y,R, G*d, Y*d, R*d] (6D)
  - Model: hidden=256, h_dim=128, z_dim=32
  - Training: 200 epochs, KL beta=0.01, cosine LR, larger batch
  - Eval: separate G→Y / Y→R / R→G transitions + dilemma zone analysis
"""
import torch, torch.nn as nn, torch.nn.functional as F
import numpy as np, os, sys, pickle, json
sys.path.insert(0, os.path.dirname(__file__))
from models import RSSM, LSTMBase, IDM, MLP, gaussian_kl

DATA = "/data/lab/swamp/data/processed"
OUT = "/data/lab/swamp/runs/p2v2"
os.makedirs(OUT, exist_ok=True)
DEV = "cuda" if torch.cuda.is_available() else "cpu"
S_DIM, A_DIM = 5, 2
PHI_DIM_NEW = 6  # [G, Y, R, G*dist, Y*dist, R*dist]
CHUNK = 100


def load_split(name):
    d = torch.load(os.path.join(DATA, f"peachtree_{name}.pt"), weights_only=False)
    sl = d["seq_lens"]
    s, a, phi, sn, valid = d["s_n"], d["a_lead_n"], d["phi"], d["s_next_n"], d["valid"]
    seqs, off = [], 0
    for L in sl:
        L = int(L)
        sq = {"s": s[off:off+L], "a": a[off:off+L], "phi": phi[off:off+L],
              "sn": sn[off:off+L], "valid": valid[off:off+L]}
        # augment signal: add distance-weighted phase features
        # s[:,0] = y_rel (normalized). dist_norm = |y_rel_norm| (already normalized)
        dist = np.abs(sq["s"][:, 0:1])  # (L,1) normalized distance
        phi_aug = np.concatenate([sq["phi"], sq["phi"] * dist], axis=1)  # (L,6)
        sq["phi"] = phi_aug.astype(np.float32)
        seqs.append(sq)
        off += L
    return seqs, d["stats"]


def chunk_seqs(seqs, chunk=CHUNK):
    out = []
    for sq in seqs:
        L = len(sq["s"])
        for i in range(0, L, chunk):
            j = min(i + chunk, L)
            if j - i < 5: break
            out.append({k: v[i:j] for k, v in sq.items()})
    return out


def pad_batch(seqs):
    B, T = len(seqs), max(len(s["s"]) for s in seqs)
    s = np.zeros((B, T, S_DIM), np.float32)
    a = np.zeros((B, T, A_DIM), np.float32)
    phi = np.zeros((B, T, PHI_DIM_NEW), np.float32)
    sn = np.zeros((B, T, S_DIM), np.float32)
    mask = np.zeros((B, T), bool)
    for i, sq in enumerate(seqs):
        L = len(sq["s"])
        s[i, :L] = sq["s"]; a[i, :L] = sq["a"]; phi[i, :L] = sq["phi"]
        sn[i, :L] = sq["sn"]; mask[i, :L] = sq["valid"]
    return (torch.tensor(s, device=DEV), torch.tensor(a, device=DEV),
            torch.tensor(phi, device=DEV), torch.tensor(sn, device=DEV),
            torch.tensor(mask, device=DEV))


def make_batches(seqs, bs=128):
    seqs = sorted(seqs, key=lambda s: len(s["s"]))
    return [seqs[i:i+bs] for i in range(0, len(seqs), bs)]


def train_model(model, seqs, epochs=200, bs=128, lr=1e-3, name="", use_kl=False, zero_phi=False):
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    chunks = chunk_seqs(seqs)
    batches = make_batches(chunks, bs)
    print(f"  {name}: {len(chunks)} chunks, {len(batches)} batches, {epochs} epochs", flush=True)
    for ep in range(epochs):
        model.train()
        np.random.shuffle(batches)
        tot, cnt = 0.0, 0
        for batch in batches:
            s, a, phi, sn, mask = pad_batch(batch)
            if zero_phi: phi = torch.zeros_like(phi)
            if use_kl:
                preds, kl = model(s, a, phi, mask)
                beta = min(0.01, ep / 20 * 0.01)  # lower beta, slower ramp
                # free bits: don't penalize KL below threshold
                kl_mean = torch.clamp(kl[mask], min=0.5).mean()
                loss = F.mse_loss(preds[mask], sn[mask]) + beta * kl_mean
            else:
                preds = model(s, a, phi, mask)
                loss = F.mse_loss(preds[mask], sn[mask])
            opt.zero_grad(); loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()
            tot += loss.item() * mask.sum().item(); cnt += mask.sum().item()
        sched.step()
        if (ep + 1) % 20 == 0:
            print(f"  {name} ep{ep+1:3d}  mse={tot/max(cnt,1):.5f}  lr={sched.get_last_lr()[0]:.6f}", flush=True)
    return model


def eval_openloop(models, test_seqs, stats, horizons=[1, 3, 5, 10]):
    s_mean, s_std = stats["s"][0], stats["s"][1]
    H = max(horizons)
    # results: model -> horizon -> dim -> {gy, yr, rg, nontrans, all}
    dims = ["y_rel", "v_f", "a_f", "gap", "dv"]
    results = {name: {h: {d: {"gy": [], "yr": [], "rg": [], "nontrans": [], "all": []}
                         for d in range(5)} for h in horizons}
               for name in models}

    for sq in test_seqs:
        L = len(sq["s"])
        if L < H + 2: continue
        s_arr, sn_arr = sq["s"], sq["sn"]
        a_arr, phi_arr = sq["a"], sq["phi"]  # phi is 6D augmented
        # original 3D phi for transition classification
        phi_orig = phi_arr[:, :3]
        valid = sq["valid"]
        for t in range(0, L - H - 1, 3):  # stride 3 for more samples
            if not valid[t]: continue
            # classify transition type
            ph = phi_orig[t:t+H+1]
            phase_ids = ph.argmax(axis=1)
            changes = np.where(np.diff(phase_ids) != 0)[0]
            cat = "nontrans"
            if len(changes) > 0:
                c = changes[0]
                from_ph, to_ph = phase_ids[c], phase_ids[c+1]
                # 0=G, 1=Y, 2=R
                if from_ph == 0 and to_ph == 1: cat = "gy"  # green→yellow
                elif from_ph == 1 and to_ph == 2: cat = "yr"  # yellow→red
                elif from_ph == 2 and to_ph == 0: cat = "rg"  # red→green
                else: cat = "nontrans"
            s0 = torch.tensor(s_arr[t:t+1], device=DEV, dtype=torch.float32)
            a_win = torch.tensor(a_arr[t:t+H], device=DEV, dtype=torch.float32).unsqueeze(0)
            phi_win = torch.tensor(phi_arr[t:t+H], device=DEV, dtype=torch.float32).unsqueeze(0)
            targets = sn_arr[t:t+H]
            for name, model in models.items():
                model.eval()
                with torch.no_grad():
                    preds = model.rollout(s0, a_win, phi_win)
                for h in horizons:
                    p = preds[0, h-1].cpu().numpy()
                    gt = targets[h-1]
                    for d in range(5):
                        e = abs(p[d] - gt[d]) * s_std[d]
                        results[name][h][d][cat].append(e)
                        results[name][h][d]["all"].append(e)
    return results


def print_results(results, horizons=[1, 3, 5, 10]):
    dims = ["y_rel(m)", "v_f(m/s)", "a_f(m/s2)", "gap(m)", "dv(m/s)"]
    for h in horizons:
        print(f"\n{'='*100}")
        print(f"Horizon {h} ({h*0.1:.1f}s) — v_f and y_rel error at transition types")
        print(f"{'='*100}")
        print(f"  {'Model':15s} | {'v_f G→Y':>10s} {'v_f Y→R':>10s} {'v_f R→G':>10s} {'v_f NonT':>10s} | {'y_rel G→Y':>12s} {'y_rel NonT':>12s}")
        for name in results:
            r = results[name][h]
            vf = r[1]  # v_f
            yr = r[0]  # y_rel
            def m(d, c): return f"{np.mean(d[c]):.4f}" if d[c] else "N/A"
            print(f"  {name:15s} | {m(vf,'gy'):>10s} {m(vf,'yr'):>10s} {m(vf,'rg'):>10s} {m(vf,'nontrans'):>10s} | {m(yr,'gy'):>12s} {m(yr,'nontrans'):>12s}")

    # signal benefit table
    print(f"\n{'='*100}")
    print("SIGNAL BENEFIT at G→Y (dilemma zone): (sig - nosig) error, negative=better")
    print(f"{'='*100}")
    for h in horizons:
        print(f"  H{h:2d}: ", end="")
        for d, dn in enumerate(dims):
            sig = results.get("RSSM-sig", {}).get(h, {}).get(d, {}).get("gy", [])
            nosig = results.get("RSSM-nosig", {}).get(h, {}).get(d, {}).get("gy", [])
            if sig and nosig:
                diff = np.mean(sig) - np.mean(nosig)
                marker = "✓" if diff < 0 else "✗"
                pct = diff / np.mean(nosig) * 100
                print(f"  {dn}: {pct:+.1f}%{marker}", end="")
            else:
                print(f"  {dn}: N/A", end="")
        print()
    # LSTM comparison
    print(f"\n  LSTM signal benefit at G→Y:")
    for h in horizons:
        print(f"  H{h:2d}: ", end="")
        for d, dn in enumerate(dims):
            sig = results.get("LSTM-sig", {}).get(h, {}).get(d, {}).get("gy", [])
            nosig = results.get("LSTM-nosig", {}).get(h, {}).get(d, {}).get("gy", [])
            if sig and nosig:
                diff = np.mean(sig) - np.mean(nosig)
                marker = "✓" if diff < 0 else "✗"
                pct = diff / np.mean(nosig) * 100
                print(f"  {dn}: {pct:+.1f}%{marker}", end="")
            else:
                print(f"  {dn}: N/A", end="")
        print()


idm = IDM()


def main():
    torch.manual_seed(42); np.random.seed(42)
    train_seqs, stats = load_split("train")
    val_seqs, _ = load_split("val")
    test_seqs, _ = load_split("test")
    print(f"Train: {len(train_seqs)}, Val: {len(val_seqs)}, Test: {len(test_seqs)}", flush=True)
    print(f"Signal dim: {PHI_DIM_NEW} (distance-weighted)", flush=True)

    models = {}
    configs = [
        ("RSSM-sig", RSSM(S_DIM, A_DIM, PHI_DIM_NEW, h_dim=128, z_dim=32, hidden=256).to(DEV), True, False),
        ("RSSM-nosig", RSSM(S_DIM, A_DIM, PHI_DIM_NEW, h_dim=128, z_dim=32, hidden=256).to(DEV), True, True),
        ("LSTM-sig", LSTMBase(S_DIM, A_DIM, PHI_DIM_NEW, hidden=128, signal=True).to(DEV), False, False),
        ("LSTM-nosig", LSTMBase(S_DIM, A_DIM, PHI_DIM_NEW, hidden=128, signal=False).to(DEV), False, True),
    ]
    for name, model, use_kl, zero_phi in configs:
        print(f"\n== Training {name} ==", flush=True)
        models[name] = train_model(model, train_seqs, epochs=200, bs=128, name=name,
                                   use_kl=use_kl, zero_phi=zero_phi)

    # calibrate IDM
    print("\n== Calibrating IDM ==", flush=True)
    s_mean, s_std = stats["s"][0], stats["s"][1]
    vf_a, gap_a, dvl_a, af_a = [], [], [], []
    for sq in chunk_seqs(train_seqs):
        v = sq["valid"]
        vf_a.append(sq["s"][v, 1] * s_std[1] + s_mean[1])
        gap_a.append(sq["s"][v, 3] * s_std[3] + s_mean[3])
        dvl_a.append(sq["s"][v, 4] * s_std[4] + s_mean[4])
        af_a.append(sq["sn"][v, 2] * s_std[2] + s_mean[2])
    idm.fit(np.concatenate(vf_a), np.concatenate(gap_a),
            np.concatenate(dvl_a), np.concatenate(af_a))
    print(f"  IDM: v0={idm.v0} T={idm.T} a={idm.a} b={idm.b} s0={idm.s0}", flush=True)

    # save
    for name, model in models.items():
        torch.save(model.state_dict(), os.path.join(OUT, f"{name}.pt"))
    pickle.dump(stats, open(os.path.join(OUT, "stats.pkl"), "wb"))
    pickle.dump((idm.v0, idm.T, idm.a, idm.b, idm.s0), open(os.path.join(OUT, "idm.pkl"), "wb"))

    # evaluate
    print("\n== Open-loop evaluation (separate transition types) ==", flush=True)
    results = eval_openloop(models, test_seqs, stats)
    print_results(results)

    # save results
    summary = {}
    for name in results:
        summary[name] = {}
        for h in results[name]:
            summary[name][str(h)] = {}
            for d in range(5):
                summary[name][str(h)][d] = {c: (float(np.mean(v)) if v else None)
                                              for c, v in results[name][h][d].items()}
    json.dump(summary, open(os.path.join(OUT, "results.json"), "w"), indent=2)
    print(f"\nResults saved to {OUT}/results.json", flush=True)


if __name__ == "__main__":
    main()
