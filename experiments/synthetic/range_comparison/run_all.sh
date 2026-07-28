#!/bin/bash
# Full experiment sequence.  Roughly twenty minutes on a laptop CPU.CPU.
set -e
cd "$(dirname "$0")"
python3 -m pytest test_rangelib.py -q
python3 verify_reference.py
python3 run_replication.py                                   # 16x16 grid, the paper's Figure 3 setting
python3 run_replication.py --topology path --nodes 33 --graphs 24 --donor-graphs 12 \
    --donors 16 --convention plain --out results/replication_path.json
python3 run_donor_sweep.py
python3 run_learned.py --steps 8000
python3 run_divergence.py
python3 run_graphlevel.py
# Follow-up on a real trained model (trains in ~2 min, then measures).
python3 ../training/marked_tree_path_graphgps.py --depths 3 --structural-channel rwse \
    --device cpu --train-graphs-per-epoch 2048 --max-epochs 60 --batch-size 64 \
    --skip-xperm-metrics --run-name range_followup --output-dir models
python3 run_realmodel.py
python3 run_beneficial.py
python3 figures.py
