"""P2: Train RSSM world model + baselines, evaluate open-loop multi-step prediction.
GO/NO-GO: does signal conditioning improve prediction at signal transitions?

Optimizations: chunk sequences to 100 steps, length-sorted batching, unbuffered output.
"""
import torch, torch.nn as nn, torch.nn.functional as F
import numpy as np, os, sys, pickle
sys.path.insert(0, os.path.dirname(__file__))
from models import RSSM, LSTMBase, IDM

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DATA = os.path.join(ROOT, "data", "processed")
OUT = os.path.join(ROOT, "runs", "p2")
os.makedirs(OUT, exist_ok=True)
DEV = "cuda" if torch.cuda.is_available() else "cpu"
S_DIM, A_DIM, PHI_DIM = 5, 2, 3
CHUNK = 100


def load_split(name):
    d = torch.load(os.path.join(DATA, f"peachtree_{name}.pt"), weights_only=False)
    sl = d["seq_lens"]
    s, a, phi, sn = d["s_n"], d["a_lead_n"], d["phi"], d["s_next_n"]
    valid = d["valid"]
    seqs, off = [], 0
    for L in sl:
        L = int(L)
        seqs.append({"s": s[off:off+L], "a": a[off:off+L], "phi": phi[off:off+L],
                     "sn": sn[off:off+L], "valid": valid[off:off+L]})
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
    phi = np.zeros((B, T, PHI_DIM), np.float32)
    sn = np.zeros((B, T, S_DIM), np.float32)
    mask = np.zeros((B, T), bool)
    for i, sq in enumerate(seqs):
        L = len(sq["s"])
        s[i, :L] = sq["s"]; a[i, :L] = sq["a"]; phi[i, :L] = sq["phi"]
        sn[i, :L] = sq["sn"]; mask[i, :L] = sq["valid"]
    return (torch.tensor(s, device=DEV), torch.tensor(a, device=DEV),
            torch.tensor(phi, device=DEV), torch.tensor(sn, device=DEV),
            torch.tensor(mask, device=DEV))


def make_batches(seqs, bs=64):
    """Sort by length, group similar-length into batches."""
    seqs = sorted(seqs, key=lambda s: len(s["s"]))
    batches = []
    for i in range(0, len(seqs), bs):
        batches.append(seqs[i:i+bs])
    return batches


def train_model(model, seqs, epochs=50, bs=64, lr=1e-3, name="", use_kl=False, zero_phi=False):
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    chunks = chunk_seqs(seqs)
    batches = make_batches(chunks, bs)
    n = sum(mask.sum().item() for _, _, _, _, mask in
            [pad_batch(b) for b in batches])
    print(f"  {name}: {len(chunks)} chunks, {len(batches)} batches", flush=True)
    for ep in range(epochs):
        model.train()
        np.random.shuffle(batches)
        tot, cnt = 0.0, 0
        for batch in batches:
            s, a, phi, sn, mask = pad_batch(batch)
            if zero_phi: phi = torch.zeros_like(phi)
            if use_kl:
                preds, kl = model(s, a, phi, mask)
                beta = min(0.1, ep / 10 * 0.1)
                loss = F.mse_loss(preds[mask], sn[mask]) + beta * kl[mask].mean()
            else:
                preds = model(s, a, phi, mask)
                loss = F.mse_loss(preds[mask], sn[mask])
            opt.zero_grad(); loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()
            tot += loss.item() * mask.sum().item(); cnt += mask.sum().item()
        if (ep + 1) % 5 == 0:
            print(f"  {name} ep{ep+1:3d}  mse={tot/max(cnt,1):.5f}", flush=True)
    return model


def eval_openloop(models, test_seqs, stats, horizons=[1, 3, 5, 10]):
    """Multi-step open-loop RMSE, split by signal transition vs non-transition."""
    s_mean, s_std = stats["s"][0], stats["s"][1]
    H = max(horizons)
    results = {name: {h: {"all": [], "trans": [], "nontrans": []} for h in horizons}
               for name in models}
    # IDM eval separately (no model rollout)
    idm_results = {h: {"all": [], "trans": [], "nontrans": []} for h in horizons}

    for sq in test_seqs:
        L = len(sq["s"])
        if L < H + 2: continue
        s_arr = sq["s"]   # normalized
        sn_arr = sq["sn"]
        a_arr = sq["a"]
        phi_arr = sq["phi"]
        valid = sq["valid"]
        # sliding windows, stride 5
        for t in range(0, L - H - 1, 5):
            if not valid[t]: continue
            # check if signal phase changes within window
            ph = phi_arr[t:t+H+1]
            phase_ids = ph.argmax(axis=1)
            is_trans = len(set(phase_ids)) > 1
            cat = "trans" if is_trans else "nontrans"
            s0 = torch.tensor(s_arr[t:t+1], device=DEV, dtype=torch.float32)
            a_win = torch.tensor(a_arr[t:t+H], device=DEV, dtype=torch.float32).unsqueeze(0)
            phi_win = torch.tensor(phi_arr[t:t+H], device=DEV, dtype=torch.float32).unsqueeze(0)
            # targets: s_next[t:t+H] (normalized)
            targets = torch.tensor(sn_arr[t:t+H], device=DEV, dtype=torch.float32).unsqueeze(0)
            for name, model in models.items():
                model.eval()
                if "RSSM" in name:
                    preds = model.rollout(s0, a_win, phi_win)  # (1,H,s_dim)
                else:
                    preds = model.rollout(s0, a_win, phi_win)
                # denormalize for interpretable RMSE
                for h in horizons:
                    p = preds[0, h-1].cpu().numpy() * s_std + s_mean
                    gt = targets[0, h-1].cpu().numpy() * s_std + s_mean
                    err = np.sqrt(np.mean((p - gt) ** 2))
                    results[name][h]["all"].append(err)
                    results[name][h][cat].append(err)
            # IDM eval
            vf = s_arr[t, 1] * s_std[1] + s_mean[1]
            for h in horizons:
                # simple IDM rollout
                v, gap, dvl = vf, s_arr[t, 3] * s_std[3] + s_mean[3], s_arr[t, 4] * s_std[4] + s_mean[4]
                for step in range(h):
                    a_idm = idm.step(v, gap, dvl)
                    v = max(0, v + a_idm * 0.1)
                    # approximate gap/dv update (simplified)
                err = abs(v - (sn_arr[t+h-1, 1] * s_std[1] + s_mean[1]))
                idm_results[h]["all"].append(err)
                idm_results[h][cat].append(err)

    # print table
    print("\n" + "=" * 80)
    print("OPEN-LOOP MULTI-STEP RMSE (denormalized, lower=better)")
    print("=" * 80)
    for h in horizons:
        print(f"\n  Horizon {h} step(s) ({h*0.1:.1f}s):")
        print(f"  {'Model':15s} {'All':>10s} {'Transition':>12s} {'NonTrans':>10s} {'Trans/NonT':>12s}")
        for name in models:
            r = results[name][h]
            a = np.mean(r["all"]) if r["all"] else 0
            t = np.mean(r["trans"]) if r["trans"] else 0
            nt = np.mean(r["nontrans"]) if r["nontrans"] else 0
            ratio = f"{t/nt:.2f}x" if nt > 0 else "N/A"
            print(f"  {name:15s} {a:10.4f} {t:12.4f} {nt:10.4f} {ratio:>12s}")
        # IDM (velocity-only RMSE)
        r = idm_results[h]
        a = np.mean(r["all"]) if r["all"] else 0
        t = np.mean(r["trans"]) if r["trans"] else 0
        nt = np.mean(r["nontrans"]) if r["nontrans"] else 0
        print(f"  {'IDM(v-only)':15s} {a:10.4f} {t:12.4f} {nt:10.4f} {'(v-only)':>12s}")
    print("=" * 80)
    return results


idm = IDM()


def main():
    torch.manual_seed(42); np.random.seed(42)
    train_seqs, stats = load_split("train")
    val_seqs, _ = load_split("val")
    test_seqs, _ = load_split("test")
    print(f"Train: {len(train_seqs)}, Val: {len(val_seqs)}, Test: {len(test_seqs)}", flush=True)

    models = {}
    configs = [
        ("RSSM-sig", RSSM(S_DIM, A_DIM, PHI_DIM).to(DEV), True, False),
        ("RSSM-nosig", RSSM(S_DIM, A_DIM, PHI_DIM).to(DEV), True, True),
        ("LSTM-sig", LSTMBase(S_DIM, A_DIM, PHI_DIM, signal=True).to(DEV), False, False),
        ("LSTM-nosig", LSTMBase(S_DIM, A_DIM, PHI_DIM, signal=False).to(DEV), False, True),
    ]
    for name, model, use_kl, zero_phi in configs:
        print(f"\n== Training {name} ==", flush=True)
        models[name] = train_model(model, train_seqs, epochs=50, name=name,
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
    pickle.dump((idm.v0, idm.T, idm.a, idm.b, idm.s0), open(os.path.join(OUT, "idm.pkl"), "wb"))
    pickle.dump(stats, open(os.path.join(OUT, "stats.pkl"), "wb"))

    # evaluate
    print("\n== Open-loop evaluation ==", flush=True)
    results = eval_openloop(models, test_seqs, stats)

    # save results
    import json
    summary = {}
    for name in models:
        summary[name] = {}
        for h in [1, 3, 5, 10]:
            r = results[name][h]
            summary[name][h] = {
                "all": float(np.mean(r["all"])) if r["all"] else 0,
                "trans": float(np.mean(r["trans"])) if r["trans"] else 0,
                "nontrans": float(np.mean(r["nontrans"])) if r["nontrans"] else 0,
                "n_trans": len(r["trans"]), "n_nontrans": len(r["nontrans"])}
    json.dump(summary, open(os.path.join(OUT, "results.json"), "w"), indent=2)
    print(f"\nResults saved to {OUT}/results.json", flush=True)


if __name__ == "__main__":
    main()
