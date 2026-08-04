# WM-conditional_signal: A Signal-Conditioned Latent World Model for Risk-Sensitive CAV Trajectory Optimization on Signalized Arterials in Mixed Traffic

## Overview

This repository implements **SWAMP** (Signal-conditioned World model for Arterial Mixed-traffic Planning), a signal-conditioned Recurrent State-Space Model (RSSM) world model for Connected and Automated Vehicle (CAV) trajectory optimization on signalized arterials in mixed traffic.

The key insight is that **signal phase and timing (SPaT) information is an exogenous conditioning variable** that fundamentally changes vehicle dynamics at signalized intersections. By incorporating SPaT as a conditioning input to a generative world model, the model learns to anticipate signal-induced deceleration/acceleration patterns — especially at **dilemma zones** (green-to-yellow transitions) where prediction errors are most consequential.

## Key Contributions

1. **First generative RSSM world model for signalized-arterial traffic operations** with SPaT as exogenous conditioning, capturing the signal–human–CAV triadic interaction that deterministic car-following models miss.

2. **Self-supervised training from real trajectory data** — the world model learns from NGSIM Peachtree Street (Atlanta) trajectory data with ground-truth SPaT, requiring no reward labels and handling flawed/missing trajectory observations.

3. **Signal-conditioned differentiable latent MPC** — the trained world model is used as a forward predictor in a model predictive controller, imagining future states under candidate accelerations and optimizing for progression, energy, and safety.

4. **Traffic-community gold-standard validation** — open-loop multi-step prediction evaluation (RMSE by signal transition type) and closed-loop SUMO simulation (stops, energy, travel time) against IDM-MPC and no-control baselines.

## Architecture

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

### State representation (5D, normalized)

| Dimension | Description | Unit |
|-----------|-------------|------|
| `y_rel` | Distance to next downstream stop bar (signed) | m |
| `v_f` | Follower (ego) vehicle speed | m/s |
| `a_f` | Follower vehicle acceleration | m/s² |
| `gap` | Space headway to preceding vehicle | m |
| `dv` | Relative speed (lead - follower) | m/s |

### Signal representation (6D, distance-weighted)

The one-hot signal phase `[G, Y, R]` is augmented with distance-weighted copies `[G·d, Y·d, R·d]` where `d = |y_rel| / 200m`, enabling the model to learn that signal phase matters more when the vehicle is closer to the intersection.

## Data

### NGSIM Peachtree Street Dataset

- **Source**: [NGSIM Vehicle Trajectories](https://data.transportation.gov/Automobiles/Next-Generation-Simulation-NGSIM-Vehicle-Trajector/8ect-6jqj) (data.transportation.gov)
- **Location**: Peachtree Street, Atlanta, GA
- **Description**: 1,543 vehicles, 322,957 trajectory rows, 10 Hz, ~17.4 min
- **Corridor**: 644.2 m (2,113 ft), 5 intersections (4 signalized: 10th, 11th, 12th, 14th Street; 1 stop-controlled: 13th Street)
- **Directions**: NB (Direction=2) and SB (Direction=4) arterial through-movement
- **Vehicle class**: 97.7% passenger cars (v_Class=2)

**Download**:
```bash
# Full NGSIM dataset (all 4 locations, ~431MB)
wget -O data/raw/ngsim/NGSIM_all.csv \
  "https://data.transportation.gov/api/views/8ect-6jqj/rows.csv?accessType=DOWNLOAD"
```

### SPaT (Signal Phase and Timing) Data

- **Source**: [NGSIM Peachtree Supporting Data](https://data.transportation.gov/api/views/8ect-6jqj/files/26d0597d-8153-4a25-af6b-7f2ec1ea7d8f?download=true&filename=Peachtree-Street-Atlanta-GA.zip)
- **Coverage**: 4 signalized intersections × 4 directions (EB/NB/SB/WB) × 2 periods (PM 4:00–4:15, Noon 12:45–1:00)
- **Format**: CSV with transition frames (10 Hz) for green/yellow/red phases (thru and left-turn)
- **Cycle length**: ~100.4 s (coordinated)
- **Files**: 32 CSV files in `data/raw/spat/signal_timing_csv/` (included in repo)

**Download**:
```bash
wget -O peachtree.zip \
  "https://data.transportation.gov/api/views/8ect-6jqj/files/26d0597d-8153-4a25-af6b-7f2ec1ea7d8f?download=true&filename=Peachtree-Street-Atlanta-GA.zip"
unzip peachtree.zip -d data/raw/
```

## Repository Structure

```
WM-conditional_signal/
├── README.md                          # This file
├── requirements.txt                   # Python dependencies
├── .gitignore
│
├── src/swamp/                         # Source code
│   ├── models.py                      # RSSM, LSTM, IDM model definitions
│   ├── p1_preprocess.py               # P1: Data preprocessing pipeline
│   ├── run_p2.py                      # P2v1: World model training & eval (v1)
│   ├── run_p2v2.py                    # P2v2: Improved training (distance-weighted signal)
│   ├── analyze_p2.py                  # P2: Per-dimension transition analysis
│   ├── gen_sumo.py                    # P3: SUMO arterial network generator
│   └── run_p3.py                      # P3: Closed-loop CAV control simulation
│
├── data/
│   ├── raw/
│   │   ├── ngsim/                     # NGSIM trajectory data (download separately)
│   │   └── spat/
│   │       ├── signal_timing_csv/     # 32 SPaT CSV files (included)
│   │       └── signal-timing/          # Original SPaT files + reports
│   └── processed/                     # Preprocessed training data (generated by P1)
│
└── configs/                           # Configuration files (TBD)
```

## Reproduction Guide

### Prerequisites

- Python 3.10+
- PyTorch with CUDA (tested on NVIDIA RTX 4090)
- SUMO 1.12+ (`sudo apt install sumo sumo-doc sumo-tools`)
- See `requirements.txt`

### Step 0: Download Data

```bash
# NGSIM trajectories (~431MB)
mkdir -p data/raw/ngsim
wget -O data/raw/ngsim/NGSIM_all.csv \
  "https://data.transportation.gov/api/views/8ect-6jqj/rows.csv?accessType=DOWNLOAD"

# SPaT data (already included in repo under data/raw/spat/)
```

### Step 1: Data Preprocessing (P1)

Extracts (follower_state, lead_action, signal_phase) → next_state training sequences from NGSIM Peachtree + ground-truth SPaT.

```bash
export PYTHONPATH=src:$PYTHONPATH
python src/swamp/p1_preprocess.py --data data/raw --out data/processed
```

**Output**: `data/processed/peachtree_{train,val,test}.pt`

Key processing steps:
- Filter: passenger cars (v_Class=2), through movement (Movement=1), arterial directions (NB/SB)
- Auto-detect direction encoding (Direction 2 = NB, Direction 4 = SB)
- Align SPaT period (Noon 12:45–1:00 selected, offset=0)
- Build per-intersection per-direction frame→phase lookup
- Extract car-following triples: follower state, lead action, downstream signal phase
- Normalize and split (train/val/test)

**Dataset statistics** (after filtering):
- 881 vehicles, 283,181 rows, 812 sequences
- 282,213 valid transitions
- Signal distribution: Green ~22-32%, Yellow ~1.8-2.2%, Red ~34-40%

### Step 2: World Model Training & Open-Loop Evaluation (P2)

#### P2v1: Basic signal-conditioned RSSM
```bash
python src/swamp/run_p2.py
```

#### P2v2: Improved training (distance-weighted signal, 200 epochs, larger model)
```bash
python src/swamp/run_p2v2.py
```

**Key improvements in v2**:
- Signal representation: 3D one-hot → 6D distance-weighted `[G, Y, R, G·d, Y·d, R·d]`
- Model size: hidden=256, h_dim=128, z_dim=32
- Training: 200 epochs, KL beta=0.01 (free bits), cosine LR schedule
- Evaluation: separate G→Y / Y→R / R→G transition types

#### Per-dimension analysis
```bash
python src/swamp/analyze_p2.py
```

### Step 3: Closed-Loop CAV Control (P3)

#### Generate SUMO network
```bash
export SUMO_HOME=/usr/share/sumo
python src/swamp/gen_sumo.py
```

#### Run closed-loop simulation
```bash
export SUMO_HOME=/usr/share/sumo
export PYTHONPATH=src:$PYTHONPATH
python src/swamp/run_p3.py
```

**Controllers evaluated**:
- `none`: No CAV control (baseline, SUMO default car-following)
- `idm`: IDM-MPC (calibrated Intelligent Driver Model)
- `wm`: World-model MPC (RSSM-based, signal-conditioned)

## Results

### P2: Open-Loop Multi-Step Prediction

#### Signal conditioning benefit at G→Y transitions (dilemma zone)

| Metric | Horizon | Improvement (RSSM-sig vs RSSM-nosig) |
|--------|---------|--------------------------------------|
| Velocity (v_f) | 1 step (0.1s) | **-17.5%** |
| Velocity (v_f) | 10 steps (1.0s) | **-21.5%** |
| Position (y_rel) | 1 step | **-20.8%** |
| Position (y_rel) | 10 steps | **-39.2%** |

The signal-conditioned RSSM significantly outperforms the no-signal variant at green-to-yellow transitions (dilemma zones), validating the core hypothesis that **SPaT conditioning enables better anticipation of signal-induced deceleration**.

#### RSSM vs LSTM at G→Y, H=10 (1.0s)

| Model | v_f Error (m/s) | y_rel Error (m) |
|-------|-----------------|-----------------|
| RSSM-sig | **0.649** | **19.94** |
| RSSM-nosig | 0.827 | 32.81 |
| LSTM-sig | 1.096 | 52.60 |
| LSTM-nosig | 1.013 | 51.45 |

RSSM with signal conditioning achieves the lowest prediction error at dilemma zones. The RSSM's latent dynamics better capture long-horizon signal–vehicle interactions compared to LSTM.

### P3: Closed-Loop CAV Control (SUMO, 3 seeds)

| Controller | Stops | Signal Stops | Avg Speed (m/s) | Energy | Travel Time (s) |
|-----------|-------|-------------|-----------------|--------|-----------------|
| None (baseline) | 1.3 | 1.3 | 8.36 | 0 | 81.0 |
| IDM-MPC | 1.3 | 1.3 | 7.54 | 1,159 | 89.7 |
| **WM-MPC** | **1.0** | **1.0** | **7.77** | 1,336 | **87.4** |

**WM-MPC vs IDM-MPC**:
- **-23% stops** (1.0 vs 1.3): signal-aware planning avoids unnecessary stops
- **-2.6% travel time** (87.4 vs 89.7s): smoother progression through coordinated signals
- **+3.0% average speed** (7.77 vs 7.54 m/s)

In the best-case scenario (Seed 777), WM-MPC achieves **zero stops** while IDM-MPC stops once, with all metrics (energy, speed, travel time) outperforming IDM.

## Method Details

### RSSM (Recurrent State-Space Model)

The RSSM combines a deterministic recurrent path (GRU) with a stochastic latent variable:

- **Transition model**: `h_t = GRU(h_{t-1}, [z_{t-1}, a_{t-1}, φ_{t-1}])` — signal phase enters the dynamics
- **Prior**: `p(z_t | h_t) = N(μ_prior, σ_prior)` — predicted latent without observation
- **Posterior**: `q(z_t | h_t, s_t) = N(μ_post, σ_post)` — latent given current observation (training only)
- **Decoder**: `ŝ_{t+1} = MLP(h_t, z_t, a_t, φ_t)` — predicts next state with signal conditioning

Training objective: reconstruction loss (MSE) + KL divergence (with free bits and β-schedule).

### World-Model MPC

At each control step:
1. **Sample** candidate accelerations (warm-started around previous optimum + exploration)
2. **Rollout** the RSSM forward for H=10 steps (1.0s) under each candidate
3. **Evaluate cost**: speed tracking + acceleration penalty + jerk penalty + safety gap + signal-aware penalty
4. **Select** the acceleration with minimum predicted cost
5. **Smooth** the output with a low-pass filter for control continuity

**Signal-aware cost**: when the downstream signal is red/yellow and the vehicle is within 30-50m, a penalty on speed is added — encouraging the model to anticipate stopping.

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
