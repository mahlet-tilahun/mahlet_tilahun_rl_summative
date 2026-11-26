"""Make all figures/tables for the report from logs/ and test the best models on unseen layouts."""
import glob
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from environment.custom_env import HealthWorkerEnv, NORTH, SOUTH, EAST, WEST, TREAT, REFER, RESTOCK, REST, POST, MAX_STOCK
from training.common import TEST_SEEDS

ALGOS = ["dqn", "reinforce", "ppo", "a2c"]
NAMES = {"dqn": "DQN", "reinforce": "REINFORCE", "ppo": "PPO", "a2c": "A2C"}
COL = {"dqn": "#1f77b4", "reinforce": "#ff7f0e", "ppo": "#2ca02c", "a2c": "#d62728"}
FIG = "assets/figures"


def load(algo):
    rows = [json.load(open(f)) for f in sorted(glob.glob(f"logs/{algo}/run_*_summary.json"))]
    return sorted(rows, key=lambda r: r["run"])


def smooth(x, w=40):
    x = np.asarray(x, float)
    if len(x) < w:
        return x
    return np.convolve(x, np.ones(w) / w, mode="valid")


def heuristic(env):
    """Hand-written 'nearest sick household' rule, used only as a reference upper-ish baseline."""
    i = env._house_at()
    if env.energy < 6: return REST
    if i is not None and env.severity[i] >= 2 and env.referrals_left > 0: return REFER
    if i is not None and env.severity[i] > 0 and env.stock > 0: return TREAT
    sick = np.where(env.severity > 0)[0]
    if env.stock == 0 or len(sick) == 0:
        tgt = np.array(POST)
    else:
        d = np.abs(env.house_pos[sick] - env.pos).sum(1) - 2 * env.severity[sick]
        tgt = env.house_pos[sick[np.argmin(d)]]
    if tuple(env.pos) == POST and env.stock == 0: return RESTOCK
    dx, dy = tgt - env.pos
    if abs(dx) >= abs(dy) and dx != 0: return EAST if dx > 0 else WEST
    if dy != 0: return NORTH if dy > 0 else SOUTH
    return REST


def run_policy(fn, seeds, use_env=False):
    env, out = HealthWorkerEnv(), []
    for s in seeds:
        obs, _ = env.reset(seed=s); tot, done = 0.0, False
        while not done:
            a = fn(env) if use_env else fn(obs)
            obs, r, te, tr, _ = env.step(a); tot += r; done = te or tr
        out.append(dict(ret=tot, stable=int((env.severity == 0).sum()), critical=int((env.severity == 3).sum()), all_stable=bool((env.severity == 0).all())))
    return out


def main():
    os.makedirs(FIG, exist_ok=True)
    data = {a: load(a) for a in ALGOS}
    best = {a: max(data[a], key=lambda r: r["mean_reward"]) for a in ALGOS}
    curves = {a: pd.read_csv(f"logs/{a}/run_{best[a]['run']:02d}_curve.csv") for a in ALGOS}

    # ---- sweep tables ----
    tables = {}
    for a in ALGOS:
        tables[a] = [dict(run=r["run"], **{k: v for k, v in r["cfg"].items() if k not in ("total_frames", "n_envs")},
                          mean_reward=round(r["mean_reward"], 2), episodes_to_converge=r["episodes_to_converge"]) for r in data[a]]

    # ---- Fig 1: cumulative reward per episode, best model of each method ----
    fig, ax = plt.subplots(1, 4, figsize=(15, 3.2), sharey=True)
    for k, a in enumerate(ALGOS):
        c = curves[a]; ax[k].plot(c.episode, c.reward, color=COL[a], alpha=0.18, lw=0.6)
        s = smooth(c.reward); ax[k].plot(np.arange(len(s)) + 40, s, color=COL[a], lw=2)
        ax[k].set_title(f"{NAMES[a]} (run {best[a]['run']})"); ax[k].set_xlabel("training episode"); ax[k].grid(alpha=0.3)
    ax[0].set_ylabel("episode return"); plt.tight_layout(); plt.savefig(f"{FIG}/fig_rewards.png", dpi=170); plt.close()

    # ---- Fig 2: stability: DQN loss + policy entropy ----
    fig, ax = plt.subplots(1, 2, figsize=(11, 3.4))
    c = curves["dqn"].dropna(subset=["loss"]); ax[0].plot(c.episode, smooth(np.r_[[c.loss.iloc[0]] * 39, c.loss.values], 40)[: len(c)], color=COL["dqn"], lw=1.8)
    ax[0].set_yscale("log"); ax[0].set_title("DQN objective (smooth-L1 TD loss, best run)"); ax[0].set_xlabel("training episode"); ax[0].set_ylabel("loss"); ax[0].grid(alpha=0.3)
    for a in ["reinforce", "ppo", "a2c"]:
        c = curves[a].dropna(subset=["entropy"]); ax[1].plot(c.episode, c.entropy.rolling(40, min_periods=1).mean(), color=COL[a], lw=1.8, label=NAMES[a])
    ax[1].axhline(np.log(8), ls="--", color="gray", lw=1); ax[1].text(5, np.log(8) - 0.12, "max entropy ln 8", color="gray", fontsize=8)
    ax[1].set_title("Policy entropy (best runs)"); ax[1].set_xlabel("training episode"); ax[1].set_ylabel("entropy (nats)"); ax[1].legend(); ax[1].grid(alpha=0.3)
    plt.tight_layout(); plt.savefig(f"{FIG}/fig_stability.png", dpi=170); plt.close()

    # ---- Fig 3: episodes to converge ----
    fig, ax = plt.subplots(1, 2, figsize=(11, 3.4))
    ax[0].bar([NAMES[a] for a in ALGOS], [best[a]["episodes_to_converge"] for a in ALGOS], color=[COL[a] for a in ALGOS])
    for k, a in enumerate(ALGOS): ax[0].text(k, best[a]["episodes_to_converge"] + 15, str(best[a]["episodes_to_converge"]), ha="center", fontsize=9)
    ax[0].set_ylabel("episodes to reach 90% of final level"); ax[0].set_title("Episodes to converge (best runs)")
    for a in ALGOS:
        c = curves[a]; s = smooth(c.reward, 50); ax[1].plot(c.frames.values[49:49 + len(s)] / 1000, s, color=COL[a], lw=1.8, label=NAMES[a])
    ax[1].set_xlabel("environment steps (thousands)"); ax[1].set_ylabel("smoothed return"); ax[1].set_title("Sample efficiency"); ax[1].legend(); ax[1].grid(alpha=0.3)
    plt.tight_layout(); plt.savefig(f"{FIG}/fig_converge.png", dpi=170); plt.close()

    # ---- Fig 4: sweep overview ----
    fig, ax = plt.subplots(1, 1, figsize=(11, 3.2)); w = 0.2
    for k, a in enumerate(ALGOS): ax.bar(np.arange(10) + (k - 1.5) * w, [r["mean_reward"] for r in data[a]], w, label=NAMES[a], color=COL[a])
    ax.set_xticks(range(10)); ax.set_xticklabels([f"run {i}" for i in range(1, 11)]); ax.set_ylabel("mean eval reward"); ax.axhline(0, color="k", lw=0.6); ax.legend(ncol=4); ax.grid(alpha=0.3, axis="y")
    ax.set_title("All 40 hyper-parameter runs (30 fixed evaluation layouts)"); plt.tight_layout(); plt.savefig(f"{FIG}/fig_sweeps.png", dpi=170); plt.close()

    # ---- Generalisation on unseen layouts ----
    from play import load_policy
    res = {}
    for a in ALGOS:
        fn, _ = load_policy(a); res[NAMES[a]] = run_policy(fn, TEST_SEEDS)
    rng = np.random.default_rng(0)
    res["Random"] = run_policy(lambda o: int(rng.integers(8)), TEST_SEEDS)
    res["Heuristic"] = run_policy(heuristic, TEST_SEEDS, use_env=True)
    summ = {k: dict(mean=float(np.mean([e["ret"] for e in v])), std=float(np.std([e["ret"] for e in v])),
                    stable=float(np.mean([e["stable"] for e in v])), critical=float(np.mean([e["critical"] for e in v])),
                    all_stable=float(np.mean([e["all_stable"] for e in v]))) for k, v in res.items()}
    fig, ax = plt.subplots(1, 2, figsize=(11, 3.4))
    ax[0].boxplot([[e["ret"] for e in v] for v in res.values()], tick_labels=list(res.keys())); ax[0].set_ylabel("return"); ax[0].set_title("Returns on 50 unseen layouts"); ax[0].grid(alpha=0.3)
    ax[1].bar(list(res.keys()), [summ[k]["stable"] for k in res], color="#4c9a2a"); ax[1].set_ylim(0, 6); ax[1].set_ylabel("households stable at end (of 6)"); ax[1].set_title("Care delivered"); ax[1].grid(alpha=0.3, axis="y")
    plt.setp(ax[1].get_xticklabels(), fontsize=8); plt.tight_layout(); plt.savefig(f"{FIG}/fig_generalization.png", dpi=170); plt.close()

    json.dump({"best": {a: dict(run=best[a]["run"], mean_reward=best[a]["mean_reward"], cfg=best[a]["cfg"], episodes_to_converge=best[a]["episodes_to_converge"]) for a in ALGOS},
               "generalization": summ, "tables": tables}, open("logs/results.json", "w"), indent=1)
    print(json.dumps(summ, indent=1))


if __name__ == "__main__":
    main()
