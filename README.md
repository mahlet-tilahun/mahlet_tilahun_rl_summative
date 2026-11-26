# Mission-Based Reinforcement Learning: Health Extension Worker in the Ethiopian Highlands

A Health Extension Worker (HEW) walks an 8×8 grid of a highland _kebele_, starting at the **Health Post**. The goal is to treat or refer sick households before **sunset**.

Medicine, energy and health-centre referral slots are limited, and untreated disease gets worse. The agent has to decide **who to visit, when to treat, when to refer, when to restock and when to rest.**

Four agents are trained with **TorchRL** on the same Gymnasium environment:

| Type            | Algorithms          |
| --------------- | ------------------- |
| Value-based     | DQN                 |
| Policy gradient | REINFORCE, PPO, A2C |

The agent's behaviour is shown live in a **three.js** 3D scene with tukuls, terraced highlands, eucalyptus and acacia trees, and Ethiopian flag colours.

## Quick start

Only [`uv`](https://docs.astral.sh/uv/) is needed.

```bash
git clone https://github.com/mahlet-tilahun/mahlet_tilahun_rl_summative.git
cd mahlet_tilahun_rl_summative
uv sync
uv run main.py
```

`uv run main.py` plays the best trained agent. Every step is printed in the terminal while the 3D scene follows it. Open the address printed in the terminal, normally http://127.0.0.1:8000/viewer.html.

## Other commands

| Command                          | What it does                                                 |
| -------------------------------- | ------------------------------------------------------------ |
| `uv run play.py --algo ppo`      | Play a specific algorithm (`dqn`, `reinforce`, `ppo`, `a2c`) |
| `uv run play.py --seed 7`        | Play a different random village                              |
| `uv run play.py --no-viewer`     | Terminal output only                                         |
| `uv run play.py --replay`        | Recorded-replay mode, without the live API                   |
| `uv run main.py train`           | Run only the experiments that have no saved result yet       |
| `uv run main.py evaluate`        | Regenerate the figures and the unseen-layout test            |
| `uv run --with pytest pytest -q` | Run the environment tests (optional)                         |

## Project layout

```
main.py                     entry point (play | train | evaluate)
play.py                     run trained agents live and serve the 3D viewer
evaluate.py                 figures and unseen-layout generalisation test
environment/
    custom_env.py           Gymnasium environment (actions, observations, rewards, terminal rules)
    rendering.py            live JSON API (/api/start, /api/step) and web server for three.js
training/
    dqn_training.py         TorchRL DQN (replay buffer, epsilon-greedy, soft target update)
    pg_training.py          TorchRL REINFORCE, PPO and A2C
    sweeps.py               10 hyperparameter runs per algorithm (40 total)
models/                     best checkpoint per algorithm (dqn/, pg/)
logs/                       learning curves (csv) and run summaries (json) for every run
assets/                     viewer.html (three.js scene), replay.js (recorded fallback), figures/
tests/                      environment unit tests
```

## The environment

**Actions (8, discrete):** North, South, East, West, Treat, Refer, Restock, Rest.

**Observation (35 values in [-1, 1]):**

- worker position, medicine stock, energy, daylight left, referral slots left, direction to the Health Post
- hints: severity at the current cell and direction to the nearest sick household
- for each of the 6 households: relative position, severity and the service needed

**Rewards:**

| Reward        | When                                         |
| ------------- | -------------------------------------------- |
| +             | treating a household (more for severe cases) |
| + bonus       | a household becomes stable                   |
| +             | a successful referral                        |
| +             | restocking when stock is low                 |
| −             | wasted actions                               |
| − small       | every step                                   |
| −             | a household becomes critical                 |
| −             | each critical household left at sunset       |
| − large       | the worker runs out of energy                |
| + large bonus | every household is stable                    |

**Episode:**

- **Start:** at the Health Post at 06:00, with full stock and energy. Households are placed at random on every reset.
- **End:** at sunset (80 steps), when all households are stable, or when the worker is exhausted.
