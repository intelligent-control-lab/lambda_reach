#!/usr/bin/env bash
# Train + test all safety-value baselines/ablations for one safety-analysis task,
# matching the existing lambda=0.99 config. Idempotent: skips a method whose
# models/last_seed_0.pt already exists (still re-runs test).
#
# Usage (from repo root, inside the IsaacLab/hj venv, in bash):
#   source /home/ruic/IsaacLab/hj/bin/activate
#   bash safety_value/scripts/2_safety_analysis/train_all_baselines.sh <safety_analysis_root>
#
# Methods (subdir names): dpe, supervised,
#   lambda_reachability_lambda_reach_lambda_0_95 / _0_50 / _0_00
# (lambda=0.99 already trained as lambda_reachability_lambda_reach_lambda_0_99.)
set -u
ROOT="${1:?usage: train_all_baselines.sh <safety_analysis_root>}"
RES="logs/safety_analysis/${ROOT}/results"
TRAIN="safety_value/scripts/2_safety_analysis/train.py"
TEST="safety_value/scripts/2_safety_analysis/test.py"

COMMON="--total_steps 10000 --eval_steps 500 --eval_batches 0 --batch 256 --lr 1e-3 --hidden_dims 256 256 --seed 0 --device cuda --epochs 20"
LAM="--max_horizon 200 --alpha_bce 5.0 --weight_main 1.0 --weight_hinge_lb 1.0 --weight_hinge_mono 1.0 --weight_bce 1.0 --target_tau_lambda 0.005 --target_update_period 1"

run() {  # $1 = model_subdir ; rest = train.py args
  local subdir="$1"; shift
  echo "==================== ${subdir} ===================="
  if [ -f "${RES}/${subdir}/models/last_seed_0.pt" ]; then
    echo "### SKIP train ${subdir} (model exists)"
  else
    echo "### TRAIN ${subdir}"
    python "${TRAIN}" --safety_analysis_root "${ROOT}" "$@" || { echo "### TRAIN FAILED ${subdir}"; return 1; }
  fi
  echo "### TEST ${subdir}"
  python "${TEST}" --safety_analysis_root "${ROOT}" --model_subdir "${subdir}" || echo "### TEST FAILED ${subdir}"
}

run dpe        --algo dpe $COMMON --lam 0.9 --lambda_update_steps 500 --lambda_max 0.99 \
               --lambda_increase_ratio 0.1 --loss_converge_tol 0.01 \
               --use_target_network --target_update_steps 10 --target_tau 0.05
run supervised --algo supervised $COMMON
run lambda_reachability_lambda_reach_lambda_0_95 --algo lambda_reachability \
               --model_tag lambda_reachability --run_name lambda_reach_lambda_0_95 $COMMON $LAM --lambda_param 0.95
run lambda_reachability_lambda_reach_lambda_0_50 --algo lambda_reachability \
               --model_tag lambda_reachability --run_name lambda_reach_lambda_0_50 $COMMON $LAM --lambda_param 0.50
run lambda_reachability_lambda_reach_lambda_0_00 --algo lambda_reachability \
               --model_tag lambda_reachability --run_name lambda_reach_lambda_0_00 $COMMON $LAM --lambda_param 0.00
echo "==================== ALL DONE (${ROOT}) ===================="
