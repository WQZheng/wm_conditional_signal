#!/bin/bash
# run_all.sh — Full reproduction pipeline for SWAMP
# Usage: bash run_all.sh [--skip-p1] [--skip-p2] [--skip-p3]
#
# Steps:
#   P1: Data preprocessing (NGSIM + SPaT -> training sequences)
#   P2: World model training + open-loop evaluation (200 epochs)
#   P3: SUMO closed-loop CAV control simulation (3 seeds x 3 controllers)
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export PYTHONPATH="$SCRIPT_DIR/src:$PYTHONPATH"
export SUMO_HOME="${SUMO_HOME:-/usr/share/sumo}"

SKIP_P1=false
SKIP_P2=false
SKIP_P3=false
for arg in "$@"; do
  case "$arg" in
    --skip-p1) SKIP_P1=true ;;
    --skip-p2) SKIP_P2=true ;;
    --skip-p3) SKIP_P3=true ;;
  esac
done

echo "============================================"
echo "  SWAMP Reproduction Pipeline"
echo "  Project root: $SCRIPT_DIR"
echo "============================================"

# --- P1: Data Preprocessing ---
if [ "$SKIP_P1" = false ]; then
  echo ""
  echo "[P1] Data Preprocessing"
  echo "--------------------------------------------"
  python "$SCRIPT_DIR/src/swamp/p1_preprocess.py" \
    --data "$SCRIPT_DIR/data/raw" \
    --out  "$SCRIPT_DIR/data/processed"
  echo "[P1] Done. Output: data/processed/peachtree_{train,val,test}.pt"
fi

# --- P2: World Model Training + Open-Loop Evaluation ---
if [ "$SKIP_P2" = false ]; then
  echo ""
  echo "[P2] World Model Training + Open-Loop Evaluation (200 epochs)"
  echo "--------------------------------------------"
  python "$SCRIPT_DIR/src/swamp/run_p2v2.py"
  echo "[P2] Done. Models saved to runs/p2v2/, results in runs/p2v2/results.json"
  echo ""
  echo "[P2] Per-dimension analysis"
  python "$SCRIPT_DIR/src/swamp/analyze_p2.py"
fi

# --- P3: SUMO Closed-Loop CAV Control ---
if [ "$SKIP_P3" = false ]; then
  echo ""
  echo "[P3] Generating SUMO arterial network"
  echo "--------------------------------------------"
  python "$SCRIPT_DIR/src/swamp/gen_sumo.py"
  echo "[P3] Network files saved to runs/p3/"

  echo ""
  echo "[P3] Closed-loop CAV control simulation (3 seeds x 3 controllers)"
  echo "--------------------------------------------"
  python "$SCRIPT_DIR/src/swamp/run_p3.py"
  echo "[P3] Done. Results saved to runs/p3/results.json"
fi

echo ""
echo "============================================"
echo "  Pipeline complete!"
echo "  Results:"
echo "    P2 (open-loop):  runs/p2v2/results.json"
echo "    P3 (closed-loop): runs/p3/results.json"
echo "============================================"
