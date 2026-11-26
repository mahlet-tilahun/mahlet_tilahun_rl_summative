"""
Hyper-parameter sweeps: 10 runs per algorithm on the SAME environment and the SAME frame budget.

    uv run python -m training.sweeps --algo all        # 40 runs (resumes automatically)
    uv run python -m training.sweeps --algo ppo
"""
import argparse
import json
import os

import torch

from training.dqn_training import train_dqn
from training.pg_training import train_pg

# every dict only lists what is CHANGED w.r.t. the defaults (run 1 = defaults)
SWEEPS = {
    "dqn": [
        {}, {"lr": 3e-4}, {"lr": 3e-3}, {"gamma": 0.95}, {"gamma": 0.999}, {"buffer_size": 10_000},
        {"batch_size": 128}, {"eps_anneal_frames": 50_000, "eps_end": 0.10}, {"eps_anneal_frames": 5_000, "eps_end": 0.02},
        {"tau_target": 0.1},
    ],
    "reinforce": [
        {}, {"lr": 3e-3}, {"lr": 3e-4}, {"gamma": 0.95}, {"frames_per_batch": 320}, {"frames_per_batch": 1280},
        {"entropy_coef": 0.05}, {"entropy_coef": 0.0}, {"normalize_returns": False}, {"hidden": 128},
    ],
    "ppo": [
        {}, {"lr": 1e-3}, {"lr": 1e-4}, {"gamma": 0.95}, {"clip_eps": 0.1}, {"clip_eps": 0.3},
        {"epochs": 8}, {"entropy_coef": 0.05}, {"entropy_coef": 0.0}, {"lmbda": 0.8},
    ],
    "a2c": [
        {}, {"lr": 3e-3}, {"lr": 3e-4}, {"gamma": 0.95}, {"frames_per_batch": 320}, {"frames_per_batch": 80},
        {"entropy_coef": 0.05}, {"entropy_coef": 0.0}, {"critic_coef": 1.0}, {"lmbda": 0.8},
    ],
}
MODEL_PATH = {"dqn": "models/dqn/dqn_best.pt", "reinforce": "models/pg/reinforce_best.pt",
              "ppo": "models/pg/ppo_best.pt", "a2c": "models/pg/a2c_best.pt"}


def run_algo(algo):
    os.makedirs("models/dqn", exist_ok=True); os.makedirs("models/pg", exist_ok=True)
    best = -1e9
    for k in range(1, 11):
        f = f"logs/{algo}/run_{k:02d}_summary.json"
        if os.path.exists(f):
            best = max(best, json.load(open(f))["mean_reward"]); continue
        cfg = SWEEPS[algo][k - 1]
        s, state = train_dqn(cfg, k, seed=k) if algo == "dqn" else train_pg(algo, cfg, k, seed=k)
        print(f"[{algo}] run {k:2d}  mean eval reward {s['mean_reward']:7.2f}  episodes {s['episodes']}  {s['seconds']:.0f}s  changed={cfg}", flush=True)
        if s["mean_reward"] > best:
            best = s["mean_reward"]
            torch.save({"state_dict": state, "cfg": s["cfg"], "run": k, "mean_reward": best}, MODEL_PATH[algo])


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--algo", default="all", choices=["all", "dqn", "reinforce", "ppo", "a2c"])
    a = ap.parse_args()
    for name in (["dqn", "reinforce", "ppo", "a2c"] if a.algo == "all" else [a.algo]):
        run_algo(name)
