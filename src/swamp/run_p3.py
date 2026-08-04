"""P3: Closed-loop CAV control with RSSM world model MPC on SUMO arterial.

CAV uses trained RSSM to imagine future states under candidate accelerations,
optimizes for progression + energy + safety. Compares with IDM-MPC and no-CAV.
"""
import os, sys, subprocess, pickle, json, random
import numpy as np
import torch

SUMO_HOME = os.environ.get("SUMO_HOME", "/usr/share/sumo")
sys.path.append(os.path.join(SUMO_HOME, "tools"))
import traci
sys.path.insert(0, os.path.dirname(__file__))
from models import RSSM, IDM

P3DIR = "/data/lab/swamp/runs/p3"
P2DIR = "/data/lab/swamp/runs/p2v2"
DEV = "cuda" if torch.cuda.is_available() else "cpu"
DT = 0.1
INT_POS = [0, 160, 320, 480, 640]  # intersection positions (m)
HORIZON = 10  # MPC horizon (steps)
N_SAMPLES = 48  # random action samples (warm-start reduces need)
V_DES = 12.0  # desired speed m/s
A_MIN, A_MAX = -5.0, 2.5
SAFE_GAP = 5.0  # minimum safe gap m
S_DIM, PHI_DIM = 5, 6
W_SPEED = 0.3
W_ACCEL = 2.0
W_JERK = 5.0
W_SAFETY = 3.0
W_SIGNAL = 0.5
ALPHA_SMOOTH = 0.6  # low-pass filter: a = alpha*prev + (1-alpha)*new


def load_model_and_stats():
    stats = pickle.load(open(os.path.join(P2DIR, "stats.pkl"), "rb"))
    s_mean, s_std = stats["s"][0], stats["s"][1]
    a_mean, a_std = stats["a_lead"][0], stats["a_lead"][1]
    model = RSSM(S_DIM, 2, PHI_DIM, h_dim=128, z_dim=32, hidden=256).to(DEV)
    model.load_state_dict(torch.load(os.path.join(P2DIR, "RSSM-sig.pt"), weights_only=True))
    model.eval()
    return model, s_mean, s_std, a_mean, a_std


def get_cav_state(vid):
    """Extract state matching training data from SUMO."""
    pos = traci.vehicle.getLanePosition(vid)  # distance along edge
    edge = traci.vehicle.getRoadID(vid)
    v = traci.vehicle.getSpeed(vid)
    a = traci.vehicle.getAcceleration(vid)
    # compute y_rel: distance to next stopbar
    # find CAV's global position along the arterial
    if edge.startswith("E"):
        edge_idx = int(edge[1:])
        global_pos = INT_POS[edge_idx] + pos
    elif edge.startswith("W"):
        edge_idx = int(edge[1:])
        global_pos = INT_POS[edge_idx + 1] - pos
    else:
        global_pos = 0  # on cross street, skip
    # next downstream intersection (NB: increasing pos)
    if edge.startswith("E"):
        next_ints = [(i, p) for i, p in enumerate(INT_POS) if p > global_pos + 2]
        next_int = min(next_ints, key=lambda x: x[1]) if next_ints else None
        if next_int:
            y_rel = next_int[1] - global_pos
            tls_id = f"N{next_int[0]}"
        else:
            y_rel = 999.0
            tls_id = None
    else:
        next_ints = [(i, p) for i, p in enumerate(INT_POS) if p < global_pos - 2]
        next_int = max(next_ints, key=lambda x: x[1]) if next_ints else None
        if next_int:
            y_rel = global_pos - next_int[1]
            tls_id = f"N{next_int[0]}"
        else:
            y_rel = 999.0
            tls_id = None
    # leader info
    try:
        leader = traci.vehicle.getLeader(vid, 200)
        if leader:
            gap = leader[1]  # gap distance
            lead_v = traci.vehicle.getSpeed(leader[0])
            dvl = lead_v - v
        else:
            gap = 200.0
            dvl = 0.0
    except traci.TraCIException:
        gap = 200.0
        dvl = 0.0
    return np.array([y_rel, v, a, gap, dvl], dtype=np.float32), tls_id, edge


def get_signal_phase(tls_id, y_rel):
    """Get 6D distance-weighted signal phase vector."""
    if tls_id is None or y_rel > 200 or y_rel < -5:
        return np.zeros(PHI_DIM, dtype=np.float32)
    state = traci.trafficlight.getRedYellowGreenState(tls_id)
    # heuristic: count green vs red characters
    n_green = sum(1 for c in state if c in "gG")
    n_yellow = sum(1 for c in state if c in "yY")
    n_red = sum(1 for c in state if c in "rR")
    total = max(n_green + n_yellow + n_red, 1)
    phi = np.zeros(3, dtype=np.float32)
    if n_yellow > 0 and n_yellow >= total * 0.1:
        phi[1] = 1.0  # yellow
    elif n_green > n_red:
        phi[0] = 1.0  # green
    else:
        phi[2] = 1.0  # red
    # distance-weighted features
    dist = abs(y_rel) / 200.0
    phi_aug = np.concatenate([phi, phi * dist]).astype(np.float32)
    return phi_aug


def mpc_wm(model, s_phys, v_lead, phi, s_mean, s_std, a_mean, a_std, prev_a=0.0):
    """World-model MPC with warm-starting. Returns best acceleration."""
    best_cost, best_a = 1e9, prev_a
    # warm-start: 80% around previous best, 20% random exploration
    n_warm = int(N_SAMPLES * 0.8)
    candidates = np.concatenate([
        np.clip(np.random.normal(prev_a, 0.5, n_warm), A_MIN, A_MAX),
        np.random.uniform(A_MIN, A_MAX, N_SAMPLES - n_warm)])
    a_lead_phys = np.array([v_lead, 0.0], dtype=np.float32)
    a_lead_norm = ((a_lead_phys - a_mean) / a_std).astype(np.float32)
    a_lead_seq = np.tile(a_lead_norm, (1, HORIZON, 1))
    phi_seq = np.tile(phi, (1, HORIZON, 1))
    a_t = torch.tensor(a_lead_seq, device=DEV, dtype=torch.float32)
    phi_t = torch.tensor(phi_seq, device=DEV, dtype=torch.float32)
    for a_cav in candidates:
        s0 = s_phys.copy()
        s0[2] = a_cav
        s0_norm = ((s0 - s_mean) / s_std).astype(np.float32)
        s0_t = torch.tensor(s0_norm, device=DEV, dtype=torch.float32).unsqueeze(0)
        with torch.no_grad():
            preds = model.rollout(s0_t, a_t, phi_t)
        preds_phys = preds[0].cpu().numpy() * s_std + s_mean
        cost = 0.0
        prev_a_h = a_cav
        for h in range(HORIZON):
            v_p, gap_p, yrel_p, a_p = preds_phys[h, 1], preds_phys[h, 3], preds_phys[h, 0], preds_phys[h, 2]
            cost += W_SPEED * (V_DES - v_p) ** 2
            cost += W_ACCEL * a_p ** 2
            cost += W_JERK * (a_p - prev_a_h) ** 2
            cost += W_SAFETY * max(0, SAFE_GAP - gap_p) ** 2
            if phi[2] > 0.5 and 0 < yrel_p < 50 and v_p > 1:
                cost += W_SIGNAL * v_p ** 2
            if phi[1] > 0.5 and 0 < yrel_p < 30 and v_p > 5:
                cost += W_SIGNAL * 1.5 * v_p ** 2
            prev_a_h = a_p
        if cost < best_cost:
            best_cost = cost
            best_a = a_cav
    return np.clip(best_a, A_MIN, A_MAX)


def run_sim(controller="wm", sim_time=300, seed=42):
    """Run SUMO simulation with CAV controller. controller: wm/idm/none."""
    random.seed(seed); np.random.seed(seed)
    cfg = os.path.join(P3DIR, "arterial.sumocfg")
    sumo_cmd = ["sumo", "-c", cfg, "--no-step-log", "--start", "--seed", str(seed)]
    traci.start(sumo_cmd)
    # randomize signal phases (force red light encounters)
    rng = random.Random(seed)
    tls_ids = list(traci.trafficlight.getIDList())
    for tid in tls_ids:
        logics = traci.trafficlight.getAllProgramLogics(tid)
        if logics:
            n_phases = len(logics[0].getPhases())
            traci.trafficlight.setPhase(tid, rng.randint(0, max(0, n_phases - 1)))
    # insert CAV at t=0
    cav_id = "cav_0"
    cav_inserted = False
    cav_departed = False
    metrics = {"stops": 0, "sig_stops": 0, "v_list": [], "a_list": [],
               "travel_time": 0, "n_vehicles": 0, "ttc_min": 999}
    was_stopped = False
    idm = IDM()
    idm.v0, idm.T, idm.a, idm.b, idm.s0 = 14, 0.8, 1.0, 3.0, 1.0
    if controller == "wm":
        model, s_mean, s_std, a_mean, a_std = load_model_and_stats()
    prev_a = 0.0
    insert_step = -1
    t = 0
    while t < sim_time * 10:
        traci.simulationStep()
        t += 1
        if not cav_inserted:
            try:
                traci.vehicle.add(vehID=cav_id, routeID="cav_route", typeID="cav",
                                  departLane="0", departPos="0", departSpeed="10")
                cav_inserted = True
                insert_step = t
            except traci.TraCIException:
                pass
        if cav_inserted and not cav_departed and t > insert_step + 2 and cav_id not in traci.vehicle.getIDList():
            cav_departed = True
            metrics["travel_time"] = t * DT
        if cav_inserted and not cav_departed and cav_id in traci.vehicle.getIDList():
            s, tls_id, edge = get_cav_state(cav_id)
            v = s[1]
            phi = get_signal_phase(tls_id, s[0])
            # record metrics
            metrics["v_list"].append(v)
            if v < 0.5 and not was_stopped:
                metrics["stops"] += 1
                if phi[2] > 0.5 or phi[1] > 0.5:
                    metrics["sig_stops"] += 1
                was_stopped = True
            elif v > 1.0:
                was_stopped = False
            # compute control
            if controller == "wm":
                v_lead = s[1] + s[4]  # lead velocity
                a_raw = mpc_wm(model, s, v_lead, phi, s_mean, s_std, a_mean, a_std, prev_a)
                a_cav = ALPHA_SMOOTH * prev_a + (1 - ALPHA_SMOOTH) * a_raw  # low-pass filter
                prev_a = a_cav
                traci.vehicle.setSpeed(cav_id, max(0, v + a_cav * DT))
                metrics["a_list"].append(a_cav)
            elif controller == "idm":
                a_idm = idm.step(s[1], s[3], s[4])
                traci.vehicle.setSpeed(cav_id, max(0, v + a_idm * DT))
                metrics["a_list"].append(a_idm)
            else:  # none
                pass  # let SUMO control the CAV with default car-following
    metrics["n_vehicles"] = len(traci.vehicle.getIDList())
    traci.close()
    metrics["avg_speed"] = np.mean(metrics["v_list"]) if metrics["v_list"] else 0
    metrics["avg_abs_accel"] = np.mean(np.abs(metrics["a_list"])) if metrics["a_list"] else 0
    metrics["energy"] = np.sum(np.array(metrics["a_list"]) ** 2) if metrics["a_list"] else 0
    return metrics


def main():
    print("== Running simulations (3 seeds) ==", flush=True)
    SEEDS = [42, 123, 777]
    all_results = {ctrl: [] for ctrl in ["none", "idm", "wm"]}
    for seed in SEEDS:
        print(f"  Seed {seed}:", flush=True)
        for ctrl in ["none", "idm", "wm"]:
            m = run_sim(controller=ctrl, seed=seed)
            all_results[ctrl].append(m)
            print(f"    {ctrl:4s}: stops={m['stops']}(sig={m['sig_stops']}), "
                  f"speed={m['avg_speed']:.2f}, accel={m['avg_abs_accel']:.3f}, "
                  f"energy={m['energy']:.1f}, tt={m['travel_time']:.1f}s", flush=True)
    # average over seeds
    print("\n== Summary (avg over 3 seeds) ==")
    print(f"  {'Controller':12s} {'Stops':>6s} {'SigStops':>9s} {'Speed':>8s} "
          f"{'Accel':>8s} {'Energy':>10s} {'Time':>8s}")
    summary = {}
    for ctrl in ["none", "idm", "wm"]:
        ms = all_results[ctrl]
        avg = {
            "stops": np.mean([m["stops"] for m in ms]),
            "sig_stops": np.mean([m["sig_stops"] for m in ms]),
            "avg_speed": np.mean([m["avg_speed"] for m in ms]),
            "avg_abs_accel": np.mean([m["avg_abs_accel"] for m in ms]),
            "energy": np.mean([m["energy"] for m in ms]),
            "travel_time": np.mean([m["travel_time"] for m in ms])}
        summary[ctrl] = avg
        print(f"  {ctrl:12s} {avg['stops']:6.1f} {avg['sig_stops']:9.1f} "
              f"{avg['avg_speed']:8.2f} {avg['avg_abs_accel']:8.3f} "
              f"{avg['energy']:10.1f} {avg['travel_time']:8.1f}")
    json.dump(summary, open(os.path.join(P3DIR, "results.json"), "w"), indent=2, default=str)
    print(f"\nResults saved to {P3DIR}/results.json")


if __name__ == "__main__":
    main()
