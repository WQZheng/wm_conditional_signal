"""
P1: Build world-model training data from NGSIM Peachtree + ground-truth SPaT.

Produces (follower_state, lead_action, signal_phase) -> next_follower_state
sequences for the signal-conditioned RSSM world model.

Pipeline:
  1. Load & filter Peachtree (cars, through, arterial dirs).
  2. Auto-detect direction code -> NB/SB and SPaT period alignment.
  3. Build per-intersection per-direction frame->phase lookup (thru G/Y/R).
  4. For each follower vehicle, join with its preceding (lead) vehicle.
  5. For each frame: next-downstream intersection -> signal phase;
     follower state (pos rel to stopbar, v, a, gap, rel-v); lead action (v,a).
  6. Build per-vehicle sequences, normalize, split train/val/test, save .pt.
"""
import argparse, os, glob, re
import numpy as np
import pandas as pd

FT2M = 0.3048
DT = 0.1  # 10 Hz

# Intersection street name -> Int_ID (south->north). 13th (Int_ID 4) is stop-controlled (no SPaT).
STREET2INT = {"10th": 1, "11th": 2, "12th": 3, "14th": 5}
# Approx stop-bar Local_Y (ft) per intersection (mid of Int_ID Local_Y range).
INT_STOPBAR_FT = {1: 180.0, 2: 700.0, 3: 1200.0, 4: 1620.0, 5: 2000.0}

PHASE_GREEN, PHASE_YELLOW, PHASE_RED = 0, 1, 2
PHASE_NAMES = ["green", "yellow", "red"]


def load_peachtree(path):
    cols = ["Vehicle_ID", "Frame_ID", "Global_Time", "Local_X", "Local_Y",
            "v_Vel", "v_Acc", "Lane_ID", "Int_ID", "Section_ID", "Direction",
            "Movement", "Preceding", "Following", "Space_Headway",
            "Time_Headway", "v_Class", "v_length", "Location"]
    df = pd.read_csv(path, usecols=cols)
    df = df[df["Location"] == "peachtree"].copy()
    # filter: passenger cars, through movement, arterial directions only
    df = df[(df["v_Class"] == 2) & (df["Movement"] == 1) & (df["Direction"].isin([2, 4]))]
    df = df.sort_values(["Vehicle_ID", "Frame_ID"]).reset_index(drop=True)
    return df


def detect_direction_mapping(df):
    """Direction code -> 'NB'/'SB' by mean d(Local_Y)/dt per vehicle."""
    out = {}
    for d in sorted(df["Direction"].unique()):
        sub = df[df["Direction"] == d]
        per_veh = sub.groupby("Vehicle_ID").apply(
            lambda g: np.polyfit(g["Frame_ID"].values, g["Local_Y"].values, 1)[0])
        slope = per_veh.mean()  # ft per frame
        out[d] = "NB" if slope > 0 else "SB"
        print(f"  Direction {d}: mean Local_Y slope={slope:.4f} ft/frame -> {out[d]}")
    return out


def parse_spat_phase_csv(path):
    """Return array of transition frames per cycle for thru phase.
    Columns: BG_Left,BY_Left,BR_Left,BG_Thru,BY_Thru,BR_Thru (frames @10Hz)."""
    s = pd.read_csv(path)
    cols = [c for c in ["BG_Thru", "BY_Thru", "BR_Thru"] if c in s.columns]
    s = s[cols].dropna()
    return s.values.astype(int)


def build_phase_lookup(transitions, n_frames):
    """Given cycle transition frames [(g_start, y_start, r_start), ...], build
    a length-n_frames int array of phase (0=green,1=yellow,2=red) for thru.
    Green: [g_start, y_start); Yellow: [y_start, r_start); Red: [r_start, next g_start)."""
    phase = np.full(n_frames, PHASE_RED, dtype=np.int8)
    cyc = transitions
    for i in range(len(cyc)):
        g, y, r = cyc[i]
        g_next = int(cyc[i + 1, 0]) if i + 1 < len(cyc) else n_frames
        if g < n_frames:
            phase[g:min(y, n_frames)] = PHASE_GREEN
        if y < n_frames:
            phase[y:min(r, n_frames)] = PHASE_YELLOW
        if r < n_frames:
            phase[r:min(g_next, n_frames)] = PHASE_RED
    return phase


def load_all_spat(spat_dir, period, dir_map):
    """Build dict[(int_id, 'NB'/'SB')] -> phase array of length n_frames.
    period: '0400-0415' or '1245-0100'. dir_map: {direction_code: 'NB'/'SB'}."""
    # determine which directions are present
    dir_codes = list(dir_map.values())  # e.g. ['NB','SB']
    spat = {}
    for street, int_id in STREET2INT.items():
        for dcode in set(dir_codes):
            f = os.path.join(spat_dir, f"Peachtree_{street}_{dcode}_{period}.csv")
            if not os.path.exists(f):
                continue
            trans = parse_spat_phase_csv(f)
            spat[(int_id, dcode)] = trans
    return spat


def align_spat_period(df, spat_dir, dir_map, n_frames):
    """Pick the SPaT period that best aligns with trajectory stop-bar stops.
    Score: correlation between (vehicle near stopbar & v<1) and (red phase).
    Try both periods and a few frame offsets; return (best_period, best_offset, score)."""
    periods = ["0400-0415", "1245-0100"]
    best = (None, 0, -1e9)
    for period in periods:
        spat = load_all_spat(spat_dir, period, dir_map)
        if not spat:
            continue
        for offset in range(0, 60, 10):  # try offsets 0..50 frames
            score = alignment_score(df, spat, dir_map, n_frames, offset)
            if score > best[2]:
                best = (period, offset, score)
            if offset == 0 and score < 0:  # negative offset direction
                for noff in range(-50, 0, 10):
                    sc = alignment_score(df, spat, dir_map, n_frames, noff)
                    if sc > best[2]:
                        best = (period, noff, sc)
    return best


def alignment_score(df, spat, dir_map, n_frames, offset):
    """Count #frames where (v<1 ft/s near a stopbar) coincides with red phase."""
    score = 0.0
    # build per (int,direction) phase arrays with offset
    phase_arr = {}
    for (int_id, dcode), trans in spat.items():
        ph = build_phase_lookup(trans, n_frames + abs(offset) + 100)
        phase_arr[(int_id, dcode)] = ph
    # sample a subset of vehicles near stopbars
    near_stop = df[(df["v_Vel"] < 1.0)].copy()
    near_stop = near_stop[near_stop["Int_ID"].isin(STREET2INT.values())]
    if len(near_stop) == 0:
        return -1e9
    sample = near_stop.sample(min(20000, len(near_stop)), random_state=0)
    for _, row in sample.iterrows():
        fid = int(row["Frame_ID"]) + offset
        dcode = dir_map[row["Direction"]]
        key = (int(row["Int_ID"]), dcode)
        if key in phase_arr and 0 <= fid < len(phase_arr[key]):
            if phase_arr[key][fid] == PHASE_RED:
                score += 1.0
            elif phase_arr[key][fid] == PHASE_GREEN:
                score -= 0.5
    return score


def next_downstream_int(local_y, direction_code, dir_map):
    """Return Int_ID of the next intersection the vehicle is approaching."""
    dcode = dir_map[direction_code]
    if dcode == "NB":  # moving +Y, next int has smallest stopbar > local_y
        cands = [(iid, sb) for iid, sb in INT_STOPBAR_FT.items() if sb > local_y + 5]
        return min(cands, key=lambda x: x[1])[0] if cands else None
    else:  # SB, moving -Y, next int has largest stopbar < local_y
        cands = [(iid, sb) for iid, sb in INT_STOPBAR_FT.items() if sb < local_y - 5]
        return max(cands, key=lambda x: x[1])[0] if cands else None


def build_sequences(df, phase_lookup, dir_map, feat_stats=None, fit_stats=True):
    """Build per-vehicle sequences of (s_t, a_lead_t, phi_t, s_{t+1}).
    phase_lookup: {(int_id, dcode): phase array}
    Returns list of dict arrays + normalization stats."""
    seqs = []
    # pivot per vehicle for lead lookup: aggregate duplicates by first
    vmap = df.groupby(["Vehicle_ID", "Frame_ID"])[["v_Vel", "v_Acc", "Local_Y"]].first()

    for vid, g in df.groupby("Vehicle_ID"):
        g = g.sort_values("Frame_ID")
        if len(g) < 5:
            continue
        fids = g["Frame_ID"].values
        dcode = dir_map[g["Direction"].iloc[0]]
        # follower states
        y = g["Local_Y"].values.astype(np.float64)
        vf = g["v_Vel"].values.astype(np.float64) * FT2M  # m/s
        af = g["v_Acc"].values.astype(np.float64) * FT2M  # m/s2
        gap = g["Space_Headway"].values.astype(np.float64) * FT2M  # m
        prec = g["Preceding"].values.astype(np.int64)
        # lead action: preceding vehicle's v,a at same frame
        vl = np.zeros_like(vf)
        al = np.zeros_like(af)
        for i, fid in enumerate(fids):
            pid = prec[i]
            if pid > 0 and (pid, fid) in vmap.index:
                lv = vmap.loc[(pid, fid)]
                if isinstance(lv, pd.DataFrame):
                    lv = lv.iloc[0]
                vl[i] = float(lv["v_Vel"]) * FT2M
                al[i] = float(lv["v_Acc"]) * FT2M
            else:
                vl[i] = vf[i]
                al[i] = af[i]
        dvl = vl - vf  # relative speed
        # signal phase: next downstream intersection's thru phase, one-hot
        phi = np.zeros((len(fids), 3), dtype=np.float32)  # G,Y,R
        y_rel = np.zeros(len(fids), dtype=np.float64)  # dist to next stopbar (m), signed (+ if before)
        for i in range(len(fids)):
            ndi = next_downstream_int(y[i] / FT2M, int(g["Direction"].iloc[0]), dir_map)
            if ndi is None:
                continue
            sb_m = INT_STOPBAR_FT[ndi] * FT2M
            if dcode == "NB":
                y_rel[i] = sb_m - (y[i] * FT2M)
            else:
                y_rel[i] = (y[i] * FT2M) - sb_m
            key = (ndi, dcode)
            if key in phase_lookup:
                ph_arr = phase_lookup[key]
                fid = int(fids[i])
                if 0 <= fid < len(ph_arr):
                    phi[i, ph_arr[fid]] = 1.0
        # assemble features: s=[y_rel, vf, af, gap, dvl], a_lead=[vl, al], phi=[G,Y,R]
        s = np.stack([y_rel, vf, af, gap, dvl], axis=1).astype(np.float32)
        a_lead = np.stack([vl, al], axis=1).astype(np.float32)
        s_next = np.roll(s, -1, axis=0)
        valid = np.zeros(len(fids), dtype=bool)
        valid[:-1] = True
        seqs.append({"s": s, "a_lead": a_lead, "phi": phi, "s_next": s_next, "valid": valid})
    return seqs


def normalize(seqs, stats=None, fit=True):
    keys = ["s", "a_lead", "s_next"]
    if fit:
        alls = {k: np.concatenate([sq[k][sq["valid"]] for sq in seqs], axis=0) for k in keys}
        stats = {k: (alls[k].mean(0), alls[k].std(0) + 1e-6) for k in keys}
    for sq in seqs:
        for k in keys:
            m, sd = stats[k]
            sq[k + "_n"] = ((sq[k] - m) / sd).astype(np.float32)
    return stats


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="/data/lab/swamp/data")
    ap.add_argument("--out", default="/data/lab/swamp/data/processed")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--val_frac", type=float, default=0.15)
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    np.random.seed(args.seed)

    print("== Loading Peachtree ==")
    df = load_peachtree(os.path.join(args.data, "ngsim_all.csv"))
    print(f"  filtered rows: {len(df)}, vehicles: {df.Vehicle_ID.nunique()}")
    n_frames = int(df.Frame_ID.max()) + 10

    print("== Detect direction mapping ==")
    dir_map = detect_direction_mapping(df)
    print(f"  dir_map: {dir_map}")

    print("== Align SPaT period ==")
    spat_dir = os.path.join(args.data, "spat", "signal_timing_csv")
    period, offset, score = align_spat_period(df, spat_dir, dir_map, n_frames)
    print(f"  best period={period}, offset={offset}, score={score:.1f}")

    print("== Build phase lookup ==")
    spat = load_all_spat(spat_dir, period, dir_map)
    phase_lookup = {}
    for (int_id, dcode), trans in spat.items():
        ph = build_phase_lookup(trans, n_frames + abs(offset) + 100)
        # apply offset by shifting
        if offset >= 0:
            ph = np.concatenate([ph[offset:], np.full(offset, PHASE_RED)])
        else:
            ph = np.concatenate([np.full(-offset, PHASE_RED), ph[:offset]])
        phase_lookup[(int_id, dcode)] = ph[:n_frames]
        # phase distribution
        u, c = np.unique(ph[:n_frames], return_counts=True)
        print(f"  Int{int_id} {dcode}: " + ", ".join(f"{PHASE_NAMES[u[j]]}={c[j]}" for j in range(len(u))))

    print("== Build sequences ==")
    seqs = build_sequences(df, phase_lookup, dir_map)
    print(f"  sequences: {len(seqs)}")
    tot = sum(sq["valid"].sum() for sq in seqs)
    print(f"  total valid transitions: {tot}")

    print("== Normalize & split ==")
    stats = normalize(seqs, fit=True)
    vids = [int(re.findall(r"\d+", str(s))[-1]) if False else i for i, s in enumerate(seqs)]
    idx = np.random.permutation(len(seqs))
    nv = int(len(seqs) * args.val_frac)
    val_idx = set(idx[:nv].tolist())
    splits = {"train": [], "val": [], "test": []}
    for i, sq in enumerate(seqs):
        if i in val_idx:
            splits["val"].append(sq)
        else:
            splits["train"].append(sq)
    # small held-out test = 10% of train
    nt = max(1, int(len(splits["train"]) * 0.10))
    splits["test"] = splits["train"][-nt:]
    splits["train"] = splits["train"][:-nt]
    for k in splits:
        print(f"  {k}: {len(splits[k])} seqs")

    import torch
    for k in splits:
        # save flat arrays for baselines
        data = {f: np.concatenate([sq[f] for sq in splits[k]], axis=0) for f in
                ["s_n", "a_lead_n", "phi", "s_next_n", "valid"]}
        # save sequence lengths for RSSM
        seq_lens = np.array([len(sq["s_n"]) for sq in splits[k]], dtype=np.int64)
        torch.save({**data, "seq_lens": seq_lens, "stats": stats},
                   os.path.join(args.out, f"peachtree_{k}.pt"))
    import pickle
    with open(os.path.join(args.out, "stats.pkl"), "wb") as f:
        pickle.dump(stats, f)
    print("== Saved to", args.out, "==")
    print("  stats:", {k: (v[0].round(3).tolist(), v[1].round(3).tolist()) for k, v in stats.items()})


if __name__ == "__main__":
    main()
