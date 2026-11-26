"""DQN (value-based) with TorchRL: replay buffer, epsilon-greedy exploration, soft-updated target network."""
import copy

import torch
from tensordict.nn import TensorDictSequential
try:
    from torchrl.collectors import Collector as SyncDataCollector   # torchrl >= 0.15
except ImportError:
    from torchrl.collectors import SyncDataCollector
from torchrl.data import LazyTensorStorage, ReplayBuffer
from torchrl.modules import EGreedyModule, QValueModule
from torchrl.objectives import DQNLoss, SoftUpdate

from training.common import (N_ACTIONS, EVAL_SEEDS, Timer, episodes_to_converge, evaluate,
                             finished_episodes, make_env, mlp, save_log, wrap)

DEFAULTS = dict(lr=1e-3, gamma=0.99, buffer_size=50_000, batch_size=64, eps_end=0.05, eps_anneal_frames=20_000,
                tau_target=0.01, hidden=64, total_frames=100_000, n_envs=4, frames_per_batch=32, updates_per_batch=8)


def build(hidden):
    env = make_env()
    qnet = wrap(mlp(N_ACTIONS, hidden), "action_value")
    qactor = TensorDictSequential(qnet, QValueModule(spec=env.action_spec))
    return env, qnet, qactor


def make_policy_fn(qnet):
    def act(obs):
        return int(qnet.module(torch.as_tensor(obs).unsqueeze(0)).argmax(-1))
    return act


def train_dqn(cfg=None, run_id=0, seed=0):
    c = {**DEFAULTS, **(cfg or {})}
    torch.manual_seed(seed)
    env, qnet, qactor = build(c["hidden"])
    train_env = make_env(c["n_envs"], seed)
    greedy = EGreedyModule(spec=env.action_spec, annealing_num_steps=c["eps_anneal_frames"], eps_init=1.0, eps_end=c["eps_end"])
    explore = TensorDictSequential(qactor, greedy)
    collector = SyncDataCollector(train_env, explore, frames_per_batch=c["frames_per_batch"], total_frames=c["total_frames"])
    rb = ReplayBuffer(storage=LazyTensorStorage(c["buffer_size"]))
    loss_fn = DQNLoss(qactor, action_space=env.action_spec, delay_value=True, loss_function="smooth_l1")
    target_updater = SoftUpdate(loss_fn, eps=1 - c["tau_target"])
    opt = torch.optim.Adam(loss_fn.parameters(), lr=c["lr"])
    loss_fn.make_value_estimator(gamma=c["gamma"])

    curve, ep_returns, frames, losses = [], [], 0, []
    with Timer() as tm:
        for batch in collector:
            frames += batch.numel()
            rb.extend(batch.reshape(-1))
            for r in finished_episodes(batch):
                ep_returns.append(r)
                curve.append(dict(episode=len(ep_returns), frames=frames, reward=r, loss=losses[-1] if losses else float("nan"), entropy=float("nan")))
            if len(rb) >= c["batch_size"]:
                for _ in range(c["updates_per_batch"]):
                    sample = rb.sample(c["batch_size"])
                    loss = loss_fn(sample)["loss"]
                    opt.zero_grad(); loss.backward()
                    torch.nn.utils.clip_grad_norm_(loss_fn.parameters(), 10.0)
                    opt.step(); target_updater.step()
                    losses.append(loss.item())
            greedy.step(batch.numel())
    collector.shutdown()
    mean_eval = float(sum(evaluate(make_policy_fn(qnet), EVAL_SEEDS)) / len(EVAL_SEEDS))
    summary = dict(algo="dqn", run=run_id, cfg=c, mean_reward=mean_eval, episodes=len(ep_returns),
                   episodes_to_converge=episodes_to_converge(ep_returns), seconds=tm.dt)
    save_log("dqn", run_id, curve, summary)
    return summary, copy.deepcopy(qnet.module.state_dict())
