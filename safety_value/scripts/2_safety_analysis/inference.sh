python safety_value/scripts/2_safety_analysis/inference.py \
    --task Isaac-Velocity-Rough-G1-PPO \
    --safety_analysis_root g1_rough_ppo_1000 \
    --model_subdir dpe_default \
    --checkpoint /home/ruic/hj_humanoid/logs/rsl_rl/g1_rough_ppo/2025-12-19_02-01-37/model_1000.pt \
    --seed 1234 \
    --video_length 1000 \
    --num_envs 1 \
    --video