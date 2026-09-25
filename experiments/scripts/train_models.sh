#!/bin/bash
# Train every model × dataset combination used in the paper.
#
# Produces one CSV per method group under logs/:
#   logs/train_tree.csv         (Xgboost, Catboost)
#   logs/train_baselines.csv    (ModernTCN, PatchTSMixer, TST, xLSTMMixer)
#   logs/train_f2f.csv
#   logs/train_isensd.csv
#   logs/train_pgd.csv
#
# Usage:  bash experiments/scripts/train_models.sh

set -e

DATASETS=(
    BeijingPM10Quality
    BeijingPM25Quality
    Panasonic18650PFData
    PPGDalia
    REVS/2013_Monterey_Motorsports_Reunion
    REVS/2013_Targa_Sixty_Six
    REVS/2014_Targa_Sixty_Six
    TennesseeEastmanProcess
    VehicleDynamicsDataset
)

run_group() {
    local out_csv="$1"; shift
    local -a models=("$@")
    local -a cfgs=()
    for ds in "${DATASETS[@]}"; do
        for m in "${models[@]}"; do
            cfgs+=( "experiments/configs/training/${ds}/${m}.yaml" )
        done
    done
    echo "=========================================="
    echo "Training [${models[*]}] -> ${out_csv}"
    echo "=========================================="
    python -m experiments.main batch --configs "${cfgs[@]}" --output "${out_csv}"
}

mkdir -p logs

run_group logs/train_tree.csv       Xgboost Catboost
run_group logs/train_baselines.csv  ModernTCN PatchTSMixer TST xLSTMMixer
run_group logs/train_f2f.csv        F2F
run_group logs/train_isensd.csv     ISensD
run_group logs/train_pgd.csv        PGD

echo "All training completed."
