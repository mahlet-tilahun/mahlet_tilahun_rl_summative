"""Policy-gradient methods with TorchRL: REINFORCE, PPO and A2C (same actor network, same environment)."""
import copy

import torch
from tensordict.nn import TensorDictModule
from torch.distributions import Categorical
try:
    from torchrl.collectors import Collector as SyncDataCollector   # torchrl >= 0.15
except ImportError:
    from torchrl.collectors import SyncDataCollector
from torchrl.data import LazyTensorStorage, ReplayBuffer, SamplerWithoutReplacement
from torchrl.modules import ProbabilisticActor, ValueOperator
from torchrl.objectives import A2CLoss, ClipPPOLoss
from torchrl.objectives.value import GAE
from torchrl.objectives.value.functional import reward2go

from training.common import (N_ACTIONS, EVAL_SEEDS, Timer, episodes_to_converge, evaluate,
                             finished_episodes, make_env, mlp, save_log, wrap)

BASE = dict(lr=1e-3, gamma=0.99, hidden=64, total_frames=120_000, n_envs=8, frames_per_batch=640, entropy_coef=0.01)
DEFAULTS = {
    "reinforce": {**BASE, "normalize_returns": True},
    "ppo": {**BASE, "lr": 5e-4, "lmbda": 0.95, "clip_eps": 0.2, "epochs": 4, "minibatch": 128, "critic_coef": 0.5},
    "a2c": {**BASE, "lmbda": 0.95, "critic_coef": 0.5, "frames_per_batch": 160},
}


def build(hidden):
    env = make_env()
    actor_net = wrap(mlp(N_ACTIONS, hidden), "logits")
    actor = ProbabilisticActor(actor_net, in_keys=["logits"], distribution_class=Categorical, spec=env.action_spec, return_log_prob=True)
    critic = ValueOperator(mlp(1, hidden), in_keys=["observation"])
    return actor, critic


def make_policy_fn(actor):
    """Greedy action (arg-max of the logits) for evaluation."""
    logits_net = actor.module[0].module
    def act(obs):
        with torch.no_grad():
            return int(logits_net(torch.as_tensor(obs).unsqueeze(0)).argmax(-1))
    return act


def train_pg(algo, cfg=None, run_id=0, seed=0):
    c = {**DEFAULTS[algo], **(cfg or {})}
    torch.manual_seed(seed)
    actor, critic = build(c["hidden"])
    train_env = make_env(c["n_envs"], seed)
    collector = SyncDataCollector(train_env, actor, frames_per_batch=c["frames_per_batch"], total_frames=c["total_frames"])
    params = list(actor.parameters()) + (list(critic.parameters()) if algo != "reinforce" else [])
    opt = torch.optim.Adam(params, lr=c["lr"])

    if algo == "ppo":
        adv = GAE(gamma=c["gamma"], lmbda=c["lmbda"], value_network=critic, average_gae=True)
        loss_fn = ClipPPOLoss(actor, critic, clip_epsilon=c["clip_eps"], entropy_coeff=c["entropy_coef"], critic_coeff=c["critic_coef"], loss_critic_type="smooth_l1")
    elif algo == "a2c":
        adv = GAE(gamma=c["gamma"], lmbda=c["lmbda"], value_network=critic, average_gae=True)
        loss_fn = A2CLoss(actor, critic, entropy_coeff=c["entropy_coef"], critic_coeff=c["critic_coef"], loss_critic_type="smooth_l1")

    curve, ep_returns, frames, last_ent, last_loss = [], [], 0, float("nan"), float("nan")
    with Timer() as tm:
        for batch in collector:
            frames += batch.numel()
            if algo == "reinforce":
                # Monte-Carlo return-to-go:  G_t = r_t + gamma * G_{t+1}
                done = batch["next", "done"]
                G = reward2go(batch["next", "reward"], done, gamma=c["gamma"])
                if c["normalize_returns"]:
                    G = (G - G.mean()) / (G.std() + 1e-8)
                dist = actor.get_dist(batch)
                logp = dist.log_prob(batch["action"])
                ent = dist.entropy().mean()
                loss = -(logp * G.squeeze(-1)).mean() - c["entropy_coef"] * ent
                opt.zero_grad(); loss.backward()
                torch.nn.utils.clip_grad_norm_(params, 1.0); opt.step()
                last_ent, last_loss = ent.item(), loss.item()
            elif algo == "a2c":
                with torch.no_grad():
                    adv(batch)
                out = loss_fn(batch)
                loss = out["loss_objective"] + out["loss_critic"] + (out["loss_entropy"] if "loss_entropy" in out.keys() else 0.0)
                opt.zero_grad(); loss.backward()
                torch.nn.utils.clip_grad_norm_(params, 0.5); opt.step()
                last_ent = out["entropy"].item() if "entropy" in out.keys() else actor.get_dist(batch).entropy().mean().item()
                last_loss = loss.item()
            else:  # ppo
                with torch.no_grad():
                    adv(batch)
                rb = ReplayBuffer(storage=LazyTensorStorage(batch.numel()), sampler=SamplerWithoutReplacement())
                rb.extend(batch.reshape(-1))
                for _ in range(c["epochs"]):
                    for _ in range(batch.numel() // c["minibatch"]):
                        sub = rb.sample(c["minibatch"])
                        out = loss_fn(sub)
                        loss = out["loss_objective"] + out["loss_critic"] + (out["loss_entropy"] if "loss_entropy" in out.keys() else 0.0)
                        opt.zero_grad(); loss.backward()
                        torch.nn.utils.clip_grad_norm_(params, 0.5); opt.step()
                last_ent = out["entropy"].item() if "entropy" in out.keys() else actor.get_dist(sub).entropy().mean().item()
                last_loss = loss.item()
            for r in finished_episodes(batch):
                ep_returns.append(r)
                curve.append(dict(episode=len(ep_returns), frames=frames, reward=r, loss=last_loss, entropy=last_ent))
    collector.shutdown()
    mean_eval = float(sum(evaluate(make_policy_fn(actor), EVAL_SEEDS)) / len(EVAL_SEEDS))
    summary = dict(algo=algo, run=run_id, cfg=c, mean_reward=mean_eval, episodes=len(ep_returns),
                   episodes_to_converge=episodes_to_converge(ep_returns), seconds=tm.dt)
    save_log(algo, run_id, curve, summary)
    return summary, copy.deepcopy(actor.state_dict())
