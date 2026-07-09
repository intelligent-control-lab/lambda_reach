#!/usr/bin/env bash
# Full real-data safety-eval pipeline refresh. Run this after ANY change to
# annotations.json, the safety-signal computation (build_from_annotations.py),
# or the trained models, so that data_raw -> inference datasets -> metrics ->
# plots all stay consistent. (Run from bash, not zsh.)
#
#   bash safety_value/scripts/refresh_real_eval.sh
#
# Stages (per real task = real_push, real_avoid):
#   1. build_from_annotations.py --write : slice annotated segments (+event onset)
#        from the demo records, compute safety_signal -> data_raw.hdf5 AND write
#        per-segment review plots to logs/safety_analysis/<root>/seg_review/.
#   2. fill_real_inference.py            : run each SIM-trained model on every
#        segment's obs -> safety_value -> results/<method>/inference/seed_0/dataset.hdf5.
# Then once, across both tasks:
#   3. evaluate_inference.py --segment_mode event_tstar : per-segment metrics
#        (clock from t_event) -> _inference_comparison/seed_0/*.csv + summary table.
#   4. plot_segment_diagnostics.py       : per-segment diagnostic figure ->
#        results/<method>/inference/seed_0/segment_diagnostics/<segment>.png.
# NOTE: no `set -u` -- the IsaacLab venv activation sources setup scripts that
# reference unbound vars (e.g. ZSH_VERSION) and would abort under nounset.
set -eo pipefail
source /home/ruic/IsaacLab/hj/bin/activate
cd /home/ruic/hj_humanoid

METHODS="lambda_reachability_lambda_reach_lambda_0_99 \
lambda_reachability_lambda_reach_lambda_0_95 \
lambda_reachability_lambda_reach_lambda_0_50 \
lambda_reachability_lambda_reach_lambda_0_00 \
dpe supervised"
SEED=0

for task in push avoid; do
  echo "=== [1] build_from_annotations ($task): data_raw.hdf5 + seg_review ==="
  python safety_value/scripts/1_data_prep/build_from_annotations.py --task "$task" --write --no_clips
  echo "=== [2] fill_real_inference ($task): per-method inference dataset.hdf5 ==="
  python safety_value/scripts/2_safety_analysis/fill_real_inference.py --task "$task" --seed "$SEED"
done

# --value_clip 1.0: clip predicted value to [-1,1] for the value-error (SVE) only, since the
# future-max target is bounded to the signal range; keeps dpe's unbounded OOD blow-ups from
# dominating SVE. Sign-based metrics (recall/FPR) are unaffected.
echo "=== [3] evaluate_inference: metrics + CSV (segment_mode=event_tstar, value_clip=1.0) ==="
python safety_value/scripts/3_result_analysis/evaluate_inference.py \
  --sa_roots real_push real_avoid --methods $METHODS --seed "$SEED" \
  --segment_mode event_tstar --value_clip 1.0

echo "=== [4] plot_segment_diagnostics: per-segment diagnostic plots ==="
python safety_value/scripts/3_result_analysis/plot_segment_diagnostics.py \
  --sa_roots real_push real_avoid --methods $METHODS --seed "$SEED" --value_clip 1.0

echo "=== DONE: real-data eval refreshed ==="
