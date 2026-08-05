"""P2 follow-up: per-dimension RMSE analysis, focusing on velocity & acceleration.
The overall RMSE is dominated by y_rel (std=190m). We need to see if signal
conditioning helps predict v, a, gap specifically at transitions.
"""
import torch, numpy as np, os, sys, pickle, json
sys.path.insert(0, os.path.dirname(__file__))
from models import RSSM, LSTMBase, IDM

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DATA = os.path.join(ROOT, "data", "processed")
OUT = os.path.join(ROOT, "runs", "p2")
DEV = "cuda" if torch.cuda.is_available() else "cpu"
S_DIM, A_DIM, PHI_DIM = 5, 2, 3
DIM_NAMES = ["y_rel(m)", "v_f(m/s)", "a_f(m/s2)", "gap(m)", "dv(m/s)"]


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


def main():
    stats = pickle.load(open(os.path.join(OUT, "stats.pkl"), "rb"))
    test_seqs, _ = load_split("test")
    s_mean, s_std = stats["s"][0], stats["s"][1]

    # load models
    models = {}
    for name, cls, kw in [
        ("RSSM-sig", RSSM, {}), ("RSSM-nosig", RSSM, {}),
        ("LSTM-sig", LSTMBase, {"signal": True}), ("LSTM-nosig", LSTMBase, {"signal": False})]:
        m = cls(S_DIM, A_DIM, PHI_DIM, **kw).to(DEV)
        m.load_state_dict(torch.load(os.path.join(OUT, f"{name}.pt"), weights_only=True))
        m.eval()
        models[name] = m

    H = 10
    horizons = [1, 3, 5, 10]
    # per-dimension per-horizon per-category errors
    errs = {name: {h: {d: {"trans": [], "nontrans": []} for d in range(5)} for h in horizons}
            for name in models}

    for sq in test_seqs:
        L = len(sq["s"])
        if L < H + 2: continue
        s_arr, sn_arr = sq["s"], sq["sn"]
        a_arr, phi_arr = sq["a"], sq["phi"]
        valid = sq["valid"]
        for t in range(0, L - H - 1, 5):
            if not valid[t]: continue
            ph = phi_arr[t:t+H+1]
            phase_ids = ph.argmax(axis=1)
            is_trans = len(set(phase_ids)) > 1
            cat = "trans" if is_trans else "nontrans"
            s0 = torch.tensor(s_arr[t:t+1], device=DEV, dtype=torch.float32)
            a_win = torch.tensor(a_arr[t:t+H], device=DEV, dtype=torch.float32).unsqueeze(0)
            phi_win = torch.tensor(phi_arr[t:t+H], device=DEV, dtype=torch.float32).unsqueeze(0)
            targets = sn_arr[t:t+H]  # (H, 5) normalized
            for name, model in models.items():
                with torch.no_grad():
                    preds = model.rollout(s0, a_win, phi_win)  # (1,H,5)
                for h in horizons:
                    p = preds[0, h-1].cpu().numpy()  # (5,) normalized
                    gt = targets[h-1]  # (5,) normalized
                    for d in range(5):
                        e = abs(p[d] - gt[d]) * s_std[d]  # denormalized per-dim
                        errs[name][h][d][cat].append(e)

    # print per-dimension tables
    for h in horizons:
        print(f"\n{'='*90}")
        print(f"Horizon {h} ({h*0.1:.1f}s) — per-dimension absolute error (denormalized)")
        print(f"{'='*90}")
        print(f"  {'Model':15s}", end="")
        for d in range(5):
            print(f" | {DIM_NAMES[d]:>20s}", end="")
        print()
        print(f"  {'':15s}", end="")
        for d in range(5):
            print(f" |  Trans  NonTrans  Gain", end="")
        print()
        print(f"  {'-'*15}", end="")
        for d in range(5):
            print(f" | {'-'*20}", end="")
        print()
        for name in models:
            print(f"  {name:15s}", end="")
            for d in range(5):
                r = errs[name][h][d]
                t = np.mean(r["trans"]) if r["trans"] else 0
                nt = np.mean(r["nontrans"]) if r["nontrans"] else 0
                print(f" | {t:6.3f}  {nt:8.3f}  {t/nt:+.2f}" if nt > 0 else f" | {t:6.3f}  {'N/A':>8s}  N/A", end="")
            print()

    # key comparison: velocity prediction at transitions
    print(f"\n{'='*90}")
    print("KEY METRIC: Velocity (v_f) prediction error at signal transitions")
    print(f"{'='*90}")
    print(f"  {'Model':15s} | {'H1 Trans':>10s} {'H1 NonT':>10s} | {'H3 Trans':>10s} {'H3 NonT':>10s} | {'H5 Trans':>10s} {'H5 NonT':>10s} | {'H10 Trans':>10s} {'H10 NonT':>10s}")
    for name in models:
        print(f"  {name:15s}", end="")
        for h in horizons:
            r = errs[name][h][1]  # v_f dimension
            t = np.mean(r["trans"]) if r["trans"] else 0
            nt = np.mean(r["nontrans"]) if r["nontrans"] else 0
            print(f" | {t:10.4f} {nt:10.4f}", end="")
        print()

    # signal conditioning benefit: RSSM-sig vs RSSM-nosig at transitions
    print(f"\n{'='*90}")
    print("SIGNAL BENEFIT: (RSSM-sig error - RSSM-nosig error) at TRANSITIONS")
    print("  Negative = signal conditioning helps (lower error)")
    print(f"{'='*90}")
    for h in horizons:
        print(f"  H{h:2d}: ", end="")
        for d in range(5):
            r_sig = errs["RSSM-sig"][h][d]["trans"]
            r_nosig = errs["RSSM-nosig"][h][d]["trans"]
            diff = np.mean(r_sig) - np.mean(r_nosig) if r_sig and r_nosig else 0
            marker = " ✓" if diff < 0 else " ✗"
            print(f" {DIM_NAMES[d]}: {diff:+.3f}{marker}", end="")
        print()

    # same for LSTM
    print(f"\n  LSTM signal benefit at transitions:")
    for h in horizons:
        print(f"  H{h:2d}: ", end="")
        for d in range(5):
            r_sig = errs["LSTM-sig"][h][d]["trans"]
            r_nosig = errs["LSTM-nosig"][h][d]["trans"]
            diff = np.mean(r_sig) - np.mean(r_nosig) if r_sig and r_nosig else 0
            marker = " ✓" if diff < 0 else " ✗"
            print(f" {DIM_NAMES[d]}: {diff:+.3f}{marker}", end="")
        print()


if __name__ == "__main__":
    main()
