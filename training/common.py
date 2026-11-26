"""Shared helpers: TorchRL environment construction, evaluation, logging."""
import json
import os
import time

import numpy as np
import torch
from tensordict.nn import TensorDictModule
from torchrl.envs import GymWrapper, SerialEnv, TransformedEnv
from torchrl.envs.transforms import RewardSum
from torchrl.modules import MLP

from environment.custom_env import HealthWorkerEnv

import warnings
warnings.filterwarnings("ignore")
os.environ.setdefault("RL_WARNINGS", "False")
torch.set_num_threads(1)
OBS_DIM = HealthWorkerEnv().obs_dim
N_ACTIONS = 8
EVAL_SEEDS = list(range(10_000, 10_030))     # never used while training
TEST_SEEDS = list(range(20_000, 20_050))     # final generalisation test (unseen layouts)


def make_env(n_envs=1, seed=0):
    """Same Gymnasium env wrapped for TorchRL (+ RewardSum -> 'episode_reward')."""
    def one():
        return TransformedEnv(GymWrapper(HealthWorkerEnv(), categorical_action_encoding=True), RewardSum())
    env = one() if n_envs == 1 else SerialEnv(n_envs, one)
    env.set_seed(seed)
    return env


def mlp(out, hidden):
    return MLP(in_features=OBS_DIM, out_features=out, num_cells=[hidden, hidden], activation_class=torch.nn.Tanh)


def wrap(net, out_key):
    return TensorDictModule(net, in_keys=["observation"], out_keys=[out_key])


@torch.no_grad()
def evaluate(act_fn, seeds):
    """Run the policy on fixed seeds with the plain Gymnasium env. Returns list of returns."""
    env = HealthWorkerEnv()
    out = []
    for s in seeds:
        obs, _ = env.reset(seed=s)
        total, done = 0.0, False
        while not done:
            obs, r, te, tr, _ = env.step(act_fn(obs))
            total += r
            done = te or tr
        out.append(total)
    return out


def finished_episodes(batch):
    """Return the episode returns of all episodes that ended inside this batch."""
    done = batch["next", "done"].squeeze(-1)
    return batch["next", "episode_reward"].squeeze(-1)[done].tolist()


def save_log(algo, run_id, curve, summary):
    d = f"logs/{algo}"
    os.makedirs(d, exist_ok=True)
    import pandas as pd
    pd.DataFrame(curve).to_csv(f"{d}/run_{run_id:02d}_curve.csv", index=False)
    with open(f"{d}/run_{run_id:02d}_summary.json", "w") as f:
        json.dump(summary, f, indent=1)


def episodes_to_converge(ep_returns, window=50, frac=0.9):
    """First episode at which the moving average reaches `frac` of its final level (relative to the start)."""
    r = np.asarray(ep_returns, dtype=float)
    if len(r) < window * 2:
        return len(r)
    ma = np.convolve(r, np.ones(window) / window, mode="valid")
    start, final = ma[0], ma[-1]
    target = start + frac * (final - start)
    hit = np.where(ma >= target)[0]
    return int(hit[0] + window) if len(hit) else len(r)


class Timer:
    def __enter__(self):
        self.t = time.time()
        return self

    def __exit__(self, *a):
        self.dt = time.time() - self.t
