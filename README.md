# WM-conditional_signal

**A Signal-Conditioned Latent World Model for Risk-Sensitive CAV Trajectory Optimization on Signalized Arterials in Mixed Traffic**

---

## Overview

This repository implements **SWAMP** (Signal-conditioned World model for Arterial Mixed-traffic Planning) — a signal-conditioned Recurrent State-Space Model (RSSM) world model for Connected and Automated Vehicle (CAV) trajectory optimization on signalized arterials in mixed traffic.

The key insight is that **signal phase and timing (SPaT) information is an exogenous conditioning variable** that fundamentally changes vehicle dynamics at signalized intersections. By incorporating SPaT as a conditioning input to a generative world model, the model learns to anticipate signal-induced deceleration/acceleration patterns — especially at **dilemma zones** (green-to-yellow transitions) where prediction errors are most consequential.

The repository includes:
- **All source code** (data preprocessing, model training, evaluation, SUMO simulation)
- **Pre-trained model checkpoints** (4 models: RSSM-sig, RSSM-nosig, LSTM-sig, LSTM-nosig)
- **Preprocessed training data** (ready to use, no need to re-run P1)
- **SUMO arterial network** (pre-generated, ready to simulate)
- **Experiment results** (P2 open-loop + P3 closed-loop)

You can clone this repo and **immediately** load the models, run evaluations, and reproduce results without training from scratch.

> **For a detailed technical description of the method, architecture, and experimental results, see [INTRODUCTION.md](INTRODUCTION.md).**

## Quick Start

```bash
# 1. Clone
git clone https://github.com/WQZheng/wm_conditional_signal.git
cd wm_conditional_signal

# 2. Install dependencies
pip install -r requirements.txt

# 3. (For P3 only) Install SUMO
sudo apt install sumo sumo-tools
export SUMO_HOME=/usr/share/sumo

# 4a. Option A — Reproduce everything from scratch (needs NGSIM data, ~30 min on GPU)
bash run_all.sh

# 4b. Option B — Skip P1 (use included preprocessed data) and P2 training,
#     only re-run evaluations and P3 simulation (~5 min):
bash run_all.sh --skip-p1

# 4c. Option C — Only re-run P3 closed-loop simulation (uses included models, ~3 min):
bash run_all.sh --skip-p1 --skip-p2
```

## Repository Structure

```
wm_conditional_signal/
├── README.md                              # This file
├── INTRODUCTION.md                         # Detailed technical description
├── requirements.txt                       # Python dependencies
├── run_all.sh                              # One-command reproduction script
├── .gitignore
│
├── src/swamp/                             # Source code
│   ├── models.py                           # RSSM, LSTM, IDM model definitions
│   ├── p1_preprocess.py                    # P1: NGSIM + SPaT -> training sequences
│   ├── run_p2.py                            # P2v1: Basic RSSM training (50 epochs, 3D signal)
│   ├── run_p2v2.py                          # P2v2: Improved training (200 epochs, 6D signal) ← best
│   ├── analyze_p2.py                        # P2: Per-dimension transition analysis
│   ├── gen_sumo.py                          # P3: SUMO arterial network generator
│   └── run_p3.py                            # P3: Closed-loop CAV control simulation
│
├── data/
│   ├── raw/
│   │   ├── ngsim/                           # NGSIM trajectories (download separately, ~431MB)
│   │   └── spat/                            # SPaT signal data (32 CSV files, included)
│   │       ├── signal_timing_csv/            #   32 CSV files (4 intersections x 4 dirs x 2 periods)
│   │       └── signal-timing/                #   Original files + PDF timing sheets
│   └── processed/                           # Preprocessed training data (included, ~23MB)
│       ├── peachtree_train.pt                #   Train split (622 sequences)
│       ├── peachtree_val.pt                  #   Val split (121 sequences)
│       ├── peachtree_test.pt                 #   Test split (69 sequences)
│       └── stats.pkl                         #   Normalization statistics
│
└── runs/                                    # Trained models & results (included)
    ├── p2/                                   # P2v1 models (50 epochs, superseded)
    │   ├── RSSM-sig.pt
    │   ├── RSSM-nosig.pt
    │   ├── LSTM-sig.pt
    │   ├── LSTM-nosig.pt
    │   ├── idm.pkl
    │   ├── stats.pkl
    │   └── results.json
    ├── p2v2/                                 # P2v2 models (200 epochs, best) ← use these
    │   ├── RSSM-sig.pt                        #   Signal-conditioned RSSM (proposed model)
    │   ├── RSSM-nosig.pt                      #   RSSM without signal (ablation)
    │   ├── LSTM-sig.pt                        #   Signal-conditioned LSTM baseline
    │   ├── LSTM-nosig.pt                      #   LSTM without signal baseline
    │   ├── idm.pkl                            #   Calibrated IDM parameters
    │   ├── stats.pkl                          #   Normalization statistics
    │   └── results.json                       #   Open-loop evaluation results
    └── p3/                                   # SUMO network + closed-loop results
        ├── arterial.net.xml                   #   SUMO network (4 intersections)
        ├── arterial.nod.xml                   #   SUMO nodes
        ├── arterial.edg.xml                   #   SUMO edges
        ├── arterial.rou.xml                   #   Traffic demand + CAV route
        ├── arterial.add.xml                   #   Signal logic
        ├── arterial.sumocfg                   #   SUMO config
        └── results.json                       #   3-seed closed-loop results
```

## Step-by-Step Reproduction

### Prerequisites

```bash
# Python 3.10+
pip install -r requirements.txt

# SUMO (for P3 closed-loop simulation only)
sudo apt install sumo sumo-tools
export SUMO_HOME=/usr/share/sumo

# Set Python path
export PYTHONPATH=$(pwd)/src:$PYTHONPATH
```

### Step 0: Download NGSIM Data (only needed for P1)

The NGSIM trajectory data is ~431MB and excluded from this repo. SPaT data is already included.

```bash
mkdir -p data/raw/ngsim
wget -O data/raw/ngsim/NGSIM_all.csv \
  "https://data.transportation.gov/api/views/8ect-6jqj/rows.csv?accessType=DOWNLOAD"
```

> **Skip this step** if using `--skip-p1` — preprocessed data is already included in `data/processed/`.

### Step 1: Data Preprocessing (P1)

Extracts (follower_state, lead_action, signal_phase) → next_follower_state training sequences from NGSIM Peachtree + ground-truth SPaT.

```bash
python src/swamp/p1_preprocess.py \
  --data data/raw \
  --out  data/processed
```

**What it does:**
1. Loads NGSIM Peachtree Street (Atlanta) trajectory data
2. Filters: passenger cars (v_Class=2), through movement (Movement=1), arterial directions (NB/SB)
3. Auto-detects direction encoding (Direction 2=NB, 4=SB) from trajectory slopes
4. Aligns SPaT period (auto-selects Noon 12:45–1:00 by matching vehicle stops with red phases)
5. Builds per-intersection per-direction frame→phase lookup (thru G/Y/R)
6. Extracts car-following triples: follower state, lead action, downstream signal phase
7. Normalizes (z-score) and splits into train/val/test

**Output:** `data/processed/peachtree_{train,val,test}.pt` + `stats.pkl`

**Dataset statistics:**
| Split | Sequences | Description |
|-------|-----------|-------------|
| Train | 622 | Used for model training |
| Val | 121 | Held out for early stopping (not used in v2) |
| Test | 69 | Held out for open-loop evaluation |
| Total | 812 | 881 vehicles, 282,213 valid transitions |

**State representation (5D, normalized):**

| Dim | Name | Description | Unit |
|-----|------|-------------|------|
| 0 | `y_rel` | Distance to next downstream stop bar (signed) | m |
| 1 | `v_f` | Follower (ego) vehicle speed | m/s |
| 2 | `a_f` | Follower vehicle acceleration | m/s² |
| 3 | `gap` | Space headway to preceding vehicle | m |
| 4 | `dv` | Relative speed (lead - follower) | m/s |

**Signal representation (6D, distance-weighted in P2v2):**

The one-hot signal phase `[G, Y, R]` is augmented with distance-weighted copies `[G·d, Y·d, R·d]` where `d = |y_rel_normalized|`, enabling the model to learn that signal phase matters more when the vehicle is closer to the intersection.

### Step 2: World Model Training + Open-Loop Evaluation (P2)

#### P2v2 (recommended — the main results in the paper)

```bash
python src/swamp/run_p2v2.py
```

**What it does:**
1. Loads preprocessed data from `data/processed/`
2. Trains 4 models for 200 epochs each:
   - **RSSM-sig**: Signal-conditioned RSSM (proposed model) — h_dim=128, z_dim=32, hidden=256
   - **RSSM-nosig**: RSSM with zeroed signal input (ablation)
   - **LSTM-sig**: Signal-conditioned LSTM baseline
   - **LSTM-nosig**: LSTM with zeroed signal input
3. Calibrates IDM on training data (grid search over v0, T, a, b, s0)
4. Evaluates open-loop multi-step prediction at horizons 1, 3, 5, 10 steps (0.1s–1.0s)
5. Separates evaluation by signal transition type: G→Y, Y→R, R→G, non-transition

**Key hyperparameters:**
| Parameter | Value |
|-----------|-------|
| Signal dim | 6 (distance-weighted) |
| Hidden dim | 256 |
| h_dim (GRU) | 128 |
| z_dim (stochastic) | 32 |
| Epochs | 200 |
| Batch size | 128 |
| Learning rate | 1e-3 (cosine annealing) |
| KL beta | 0.01 (with free bits=0.5) |
| Gradient clip | 5.0 |
| Chunk length | 100 steps |

**Output:** Model checkpoints in `runs/p2v2/`, results in `runs/p2v2/results.json`

#### P2v1 (basic version, superseded)

```bash
python src/swamp/run_p2.py
```

Trains with 3D one-hot signal, 50 epochs, smaller model (h_dim=64, z_dim=16, hidden=128).

#### Per-dimension analysis

```bash
python src/swamp/analyze_p2.py
```

Prints per-dimension RMSE breakdown (y_rel, v_f, a_f, gap, dv) for all models.

### Step 3: SUMO Closed-Loop CAV Control (P3)

#### Generate SUMO network

```bash
python src/swamp/gen_sumo.py
```

Generates a 4-intersection arterial network (640m total, 160m spacing) with:
- 5 nodes (4 signalized intersections + endpoints)
- 2-lane arterial edges (NB/SB) + 1-lane cross streets
- Coordinated fixed-time signals (100s cycle, 40s green, 4s yellow)
- Traffic demand: NB flow (p=0.2), SB flow (p=0.15), cross-street flow (p=0.05)
- CAV route: E0→E1→E2→E3 (full arterial traversal)

**Output:** `runs/p3/arterial.{net,nod,edg,rou,add,sumocfg}.xml`

#### Run closed-loop simulation

```bash
python src/swamp/run_p3.py
```

**What it does:**
1. Loads the pre-trained RSSM-sig model from `runs/p2v2/`
2. For each of 3 random seeds (42, 123, 777):
   - Randomizes signal initial phases (to force red-light encounters)
   - Inserts a CAV at t=0 on the arterial route
   - Runs 300s simulation with 3 controllers:
     - **none**: No CAV control (SUMO default car-following baseline)
     - **idm**: IDM-MPC (calibrated Intelligent Driver Model)
     - **wm**: World-model MPC (RSSM-based, signal-conditioned)
3. Averages metrics over seeds

**WM-MPC algorithm:**
1. At each 0.1s control step, sample 48 candidate accelerations (80% warm-started around previous optimum, 20% random exploration)
2. Roll out the RSSM forward for H=10 steps (1.0s) under each candidate
3. Evaluate cost: speed tracking + acceleration + jerk + safety gap + signal-aware penalty
4. Select minimum-cost acceleration, smooth with low-pass filter (α=0.6)

**MPC cost function:**
```
J = Σ_h [ W_SPEED·(v_des - v)² + W_ACCEL·a² + W_JERK·(a - a_prev)²
        + W_SAFETY·max(0, safe_gap - gap)² + W_SIGNAL·signal_penalty ]
```

| Weight | Value | Description |
|--------|-------|-------------|
| W_SPEED | 0.3 | Speed tracking (desired 12 m/s) |
| W_ACCEL | 2.0 | Acceleration penalty |
| W_JERK | 5.0 | Jerk (acceleration change) penalty |
| W_SAFETY | 3.0 | Safety gap violation (min 5m) |
| W_SIGNAL | 0.5 | Signal-aware speed penalty (red < 50m, yellow < 30m) |

**Output:** `runs/p3/results.json`

## Using Pre-Trained Models

The repository includes all trained model checkpoints. You can load them directly:

```python
import torch, pickle, sys, os
sys.path.insert(0, "src/swamp")
from models import RSSM

# Load normalization stats
stats = pickle.load(open("runs/p2v2/stats.pkl", "rb"))
s_mean, s_std = stats["s"][0], stats["s"][1]

# Load the signal-conditioned RSSM (proposed model)
model = RSSM(s_dim=5, a_dim=2, phi_dim=6, h_dim=128, z_dim=32, hidden=256)
model.load_state_dict(torch.load("runs/p2v2/RSSM-sig.pt", weights_only=True))
model.eval()

# Open-loop multi-step prediction
# s0: (1, 5) normalized initial state
# a_seq: (1, H, 2) normalized lead actions
# phi_seq: (1, H, 6) distance-weighted signal phases
preds = model.rollout(s0, a_seq, phi_seq)  # (1, H, 5) normalized predictions

# Denormalize
preds_phys = preds.cpu().numpy() * s_std + s_mean
```

### Available checkpoints

| Model | File | Size | Description |
|-------|------|------|-------------|
| RSSM-sig | `runs/p2v2/RSSM-sig.pt` | 1.1 MB | Signal-conditioned RSSM (proposed, 200 epochs) |
| RSSM-nosig | `runs/p2v2/RSSM-nosig.pt` | 1.1 MB | RSSM without signal (ablation) |
| LSTM-sig | `runs/p2v2/LSTM-sig.pt` | 357 KB | Signal-conditioned LSTM baseline |
| LSTM-nosig | `runs/p2v2/LSTM-nosig.pt` | 345 KB | LSTM without signal baseline |
| RSSM-sig (v1) | `runs/p2/RSSM-sig.pt` | 264 KB | Earlier version (50 epochs, 3D signal) |
| IDM | `runs/p2v2/idm.pkl` | 53 B | Calibrated IDM parameters (v0, T, a, b, s0) |

## Results

### P2: Open-Loop Multi-Step Prediction

#### Signal conditioning benefit at G→Y transitions (dilemma zone)

The core hypothesis: SPaT conditioning should improve prediction at green-to-yellow transitions, where drivers face stop-or-go decisions.

| Metric | Horizon | RSSM-sig | RSSM-nosig | Improvement |
|--------|---------|----------|------------|-------------|
| Velocity (v_f) | 1 step (0.1s) | 0.970 m/s | 1.176 m/s | **-17.5%** |
| Velocity (v_f) | 10 steps (1.0s) | 0.649 m/s | 0.827 m/s | **-21.5%** |
| Position (y_rel) | 1 step | — | — | **-20.8%** |
| Position (y_rel) | 10 steps | — | — | **-39.2%** |

#### RSSM vs LSTM at G→Y, H=10 (1.0s horizon)

| Model | v_f Error (m/s) | y_rel Error (m) |
|-------|-----------------|-----------------|
| **RSSM-sig** | **0.649** | **19.94** |
| RSSM-nosig | 0.827 | 32.81 |
| LSTM-sig | 1.096 | 52.60 |
| LSTM-nosig | 1.013 | 51.45 |

**Key finding:** RSSM with signal conditioning achieves the lowest prediction error at dilemma zones. The signal benefit grows with horizon for RSSM (long-horizon signal–vehicle interactions) but not for LSTM.

### P3: Closed-Loop CAV Control (SUMO, 3 seeds averaged)

| Controller | Stops | Signal Stops | Avg Speed (m/s) | Energy | Travel Time (s) |
|-----------|-------|-------------|-----------------|--------|-----------------|
| None (baseline) | 1.3 | 1.3 | 8.36 | 0 | 81.0 |
| IDM-MPC | 1.3 | 1.3 | 7.54 | 1,159 | 89.7 |
| **WM-MPC** | **1.0** | **1.0** | **7.77** | 1,336 | **87.4** |

**WM-MPC vs IDM-MPC:**
- **-23% stops** (1.0 vs 1.3): signal-aware planning avoids unnecessary stops
- **-2.6% travel time** (87.4 vs 89.7s): smoother progression through coordinated signals
- **+3.0% average speed** (7.77 vs 7.54 m/s)
- Energy +15% (acceptable trade-off for fewer stops)

**Best-case (Seed 777):** WM-MPC achieves **zero stops** while IDM-MPC stops once, with all metrics (energy, speed, travel time) outperforming IDM.

## Architecture

### RSSM (Recurrent State-Space Model)

```
                    ┌──────────────────────────────────────────┐
                    │     Signal-Conditioned RSSM World Model     │
                    │                                          │
  s_t ──────────────┤──► Posterior q(z|s,h) ──► z_t            │
  [y_rel, v, a,     │                          (stochastic)      │
   gap, dv]         │                              │            │
                    │    ┌─── GRU ───┐             │            │
  a_t ──────────────┼──►│ h_t = GRU │◄── z_{t-1} ─┘            │
  [v_lead, a_lead]  │    │  (h, [z,  │                          │
                    │    │   a, φ])  │             ┌──────────┐ │
  φ_t ──────────────┼──►│           │──── h_t ──► │ Decoder  │ │──► ŝ_{t+1}
  [G, Y, R,         │    └───────────┘             │ (h,z,a,φ)│ │
   G·d, Y·d, R·d]   │                              └──────────┘ │
                    │  Prior p(z|h)                  │           │
                    └──────────────────────────────┼───────────┘
                                                   │
                                           ┌───────▼───────┐
                                           │  Latent MPC    │
                                           │  (CVaR risk)   │
                                           │  minimize cost │
                                           └───────────────┘
```

- **Transition model**: `h_t = GRU(h_{t-1}, [z_{t-1}, a_{t-1}, φ_{t-1}])` — signal phase enters the dynamics
- **Prior**: `p(z_t | h_t) = N(μ_prior, σ_prior)` — predicted latent without observation
- **Posterior**: `q(z_t | h_t, s_t) = N(μ_post, σ_post)` — latent given current observation (training only)
- **Decoder**: `ŝ_{t+1} = MLP(h_t, z_t, a_t, φ_t)` — predicts next state with signal conditioning

Training objective: reconstruction loss (MSE) + KL divergence (with free bits and β-schedule).

## Data Sources

| Data | Source | Size | License |
|------|--------|------|---------|
| NGSIM Trajectories | [data.transportation.gov](https://data.transportation.gov/Automobiles/Next-Generation-Simulation-NGSIM-Vehicle-Trajector/8ect-6jqj) | ~431 MB | Public Domain |
| NGSIM SPaT | [Peachtree Supporting Data](https://data.transportation.gov/api/views/8ect-6jqj/files/26d0597d-8153-4a25-af6b-7f2ec1ea7d8f) | ~373 KB | Public Domain |

## Environment

Tested on:
- Ubuntu 22.04, Linux kernel 6.8.0
- NVIDIA RTX 4090 (24 GB), CUDA 12.x
- Python 3.10.12, PyTorch 2.x
- SUMO 1.12.0
- 256 CPU cores, 976 GB RAM

## Citation

If you use this code or data, please cite:

```bibtex
@article{swamp2026,
  title={A Signal-Conditioned Latent World Model for Risk-Sensitive CAV Trajectory Optimization on Signalized Arterials in Mixed Traffic},
  author={Zheng, W.},
  journal={Transportation Research Part C: Emerging Technologies},
  year={2026},
  note={Under review}
}
```

## License

This project is for research purposes. The NGSIM dataset is in the public domain (U.S. Government work).
