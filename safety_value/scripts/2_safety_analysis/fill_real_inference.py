#!/usr/bin/env python3
"""
Build per-method inference dataset.hdf5 for the real tasks, so evaluate_inference.py can run.

For each method, load its SIM-trained safety value, run it on each real segment's policy_obs to
produce safety_value[T], and write:
  logs/safety_analysis/<real_root>/results/<method>/inference/seed_<seed>/dataset.hdf5
with per-episode {safety_signal_*, event_*, safety_value}. (Segments + event flags come from
the annotation-built data_raw.hdf5.)

Run in bash inside the IsaacLab/hj venv.
"""
import argparse, json, os, sys
from pathlib import Path
import numpy as np, torch, h5py

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
from safety_value.safety_analysis_algos.model import build_mlp

TASKS = {
    "push":  dict(sim="g1_29dof_flat_unitree_ppo_6000",  real="real_push",
                  sig="safety_signal_balance",   evt="event_push"),
    "avoid": dict(sim="g1_collision_avoid_fwd_back_4999", real="real_avoid",
                  sig="safety_signal_collision", evt="event_ball_spawn"),
}
METHODS = [
    "lambda_reachability_lambda_reach_lambda_0_99",
    "lambda_reachability_lambda_reach_lambda_0_95",
    "lambda_reachability_lambda_reach_lambda_0_50",
    "lambda_reachability_lambda_reach_lambda_0_00",
    "dpe", "supervised",
]

def load_value_fn(sim_root, method, device):
    mdir = f"logs/safety_analysis/{sim_root}/results/{method}"
    cfg = json.load(open(f"{mdir}/training_config.json"))
    algo = cfg.get("algorithm", "dpe"); hidden = tuple(cfg.get("hidden_dims", [256, 256])); idim = cfg["input_dim"]
    ck = torch.load(f"{mdir}/models/last_seed_0.pt", map_location=device)
    if algo == "lambda_reachability":
        c1 = build_mlp(input_dim=idim, hidden_dims=hidden).to(device)
        c2 = build_mlp(input_dim=idim, hidden_dims=hidden).to(device)
        c1.load_state_dict(ck["critic1"]); c2.load_state_dict(ck["critic2"]); c1.eval(); c2.eval()
        def fn(x):
            with torch.no_grad(): return (0.5 * (c1(x) + c2(x))).view(-1)
    else:
        m = build_mlp(input_dim=idim, hidden_dims=hidden).to(device)
        m.load_state_dict(ck["model_state"]); m.eval()
        def fn(x):
            with torch.no_grad(): return m(x).view(-1)
    return fn, idim, algo

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", choices=["push", "avoid"], required=True)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--methods", nargs="+", default=METHODS)
    args = ap.parse_args()
    t = TASKS[args.task]
    device = "cuda" if torch.cuda.is_available() else "cpu"
    raw_path = f"logs/safety_analysis/{t['real']}/data_raw.hdf5"
    with h5py.File(raw_path, "r") as raw:
        eps = list(raw["data"].keys())
        data = {ep: dict(obs=np.asarray(raw["data"][ep]["policy_obs"], np.float32),
                         sig=np.asarray(raw["data"][ep][t["sig"]], np.float32),
                         evt=np.asarray(raw["data"][ep][t["evt"]], np.float32)) for ep in eps}
    print(f"[{args.task}] {len(eps)} segments from {raw_path}")
    for method in args.methods:
        fn, idim, algo = load_value_fn(t["sim"], method, device)
        out = f"logs/safety_analysis/{t['real']}/results/{method}/inference/seed_{args.seed}"
        os.makedirs(out, exist_ok=True)
        with h5py.File(f"{out}/dataset.hdf5", "w") as h:
            g = h.create_group("data")
            for ep, d in data.items():
                assert d["obs"].shape[1] == idim, f"obs dim {d['obs'].shape[1]} != model {idim}"
                sv = fn(torch.as_tensor(d["obs"], device=device)).cpu().numpy().astype(np.float32)
                ge = g.create_group(ep)
                ge.create_dataset(t["sig"], data=d["sig"])
                ge.create_dataset(t["evt"], data=d["evt"])
                ge.create_dataset("safety_value", data=sv)
        print(f"  {method:48s} ({algo}) -> {out}/dataset.hdf5")
    print("done")

if __name__ == "__main__":
    main()
