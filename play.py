"""
play.py - run the trained agents in the Health Extension Worker environment.

    uv run play.py                  # LIVE: the best agent plays while the 3D scene follows; every step is printed here
    uv run play.py --algo ppo --seed 7
    uv run play.py --no-viewer      # terminal only (full episode printed at once)
    uv run play.py --replay         # old mode: record all four agents first, then replay them in 3D
"""
import argparse
import json
import os

import torch

from environment.custom_env import ACTION_NAMES, HealthWorkerEnv
from environment.rendering import LiveSession, record_episode, save_replay, serve_viewer

ALGOS = ["dqn", "reinforce", "ppo", "a2c"]
PATHS = {"dqn": "models/dqn/dqn_best.pt", "reinforce": "models/pg/reinforce_best.pt",
         "ppo": "models/pg/ppo_best.pt", "a2c": "models/pg/a2c_best.pt"}


def load_policy(algo):
    """Rebuild the TorchRL network from the saved checkpoint and return (act_fn, checkpoint)."""
    ck = torch.load(PATHS[algo], weights_only=False)
    if algo == "dqn":
        from training.dqn_training import build, make_policy_fn
        _, qnet, _ = build(ck["cfg"]["hidden"])
        qnet.module.load_state_dict(ck["state_dict"])
        return make_policy_fn(qnet), ck
    from training.pg_training import build, make_policy_fn
    actor, _ = build(ck["cfg"]["hidden"])
    actor.load_state_dict(ck["state_dict"])
    return make_policy_fn(actor), ck


def step_line(env, a, r):
    return (f"t={env.t:2d}  {ACTION_NAMES[a]:8s} r={r:6.2f}  stock={env.stock} energy={env.energy:2d} "
            f"referrals={env.referrals_left}  severity={[int(s) for s in env.severity]}")


def summary_line(env, total):
    return (f"--> episode return {total:.1f} | stable households {int((env.severity == 0).sum())}/{len(env.severity)} | "
            f"critical left {int((env.severity == 3).sum())}")


def verbose_episode(env, act, seed, name):
    obs, _ = env.reset(seed=seed)
    print(f"\n=== {name} | seed {seed} ===")
    print(env.render())
    total, done = 0.0, False
    while not done:
        a = act(obs)
        obs, r, te, tr, _ = env.step(a)
        total += r
        done = te or tr
        print(step_line(env, a, r))
    print(summary_line(env, total))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--algo", default="best", choices=["best"] + ALGOS)
    ap.add_argument("--seed", type=int, default=2026)
    ap.add_argument("--no-viewer", action="store_true")
    ap.add_argument("--replay", action="store_true")
    ap.add_argument("--port", type=int, default=8000)
    args = ap.parse_args()

    missing = [a for a in ALGOS if not os.path.exists(PATHS[a])]
    if missing:
        raise SystemExit(f"Missing trained models for {missing}. Run: uv run main.py train")

    pols = {a: load_policy(a) for a in ALGOS}
    score = {a: pols[a][1]["mean_reward"] for a in ALGOS}
    best = max(score, key=score.get) if args.algo == "best" else args.algo
    print("Sweep scores (mean eval reward):", {a: round(v, 1) for a, v in score.items()}, "-> playing:", best.upper())
    env = HealthWorkerEnv()

    if args.no_viewer:
        verbose_episode(env, pols[best][0], args.seed, best.upper())
        return

    if args.replay:                       # recorded replay (works without the live API)
        verbose_episode(env, pols[best][0], args.seed, best.upper())
        order = [best] + [a for a in ALGOS if a != best]
        eps = [record_episode(env, pols[a][0], seed=args.seed, label=a.upper() + (" ★" if a == best else "")) for a in order]
        print("Saved replay ->", save_replay(eps))
        serve_viewer("assets", args.port)
        return

    # ---- LIVE mode: the browser asks for one step at a time ----
    def on_start(algo, seed, e):
        print(f"\n=== LIVE: {algo.upper()} | village seed {seed} ===")
        print(e.render())

    def on_step(e, a, r, done, total):
        print(step_line(e, a, r))
        if done:
            print(summary_line(e, total))

    session = LiveSession(env, {a: pols[a][0] for a in ALGOS}, best, args.seed, on_start, on_step)
    serve_viewer("assets", args.port, session=session)


if __name__ == "__main__":
    main()
