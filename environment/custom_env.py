"""
HealthWorkerEnv - a Gymnasium environment for an Ethiopian Health Extension Worker (HEW).

Mission: one HEW starts at the village Health Post, walks on an 8x8 grid of the highland
kebele, and must treat sick households before sunset.  Disease gets worse over time, medicine
and energy are limited, and the health centre only accepts 2 referrals per day.  So the agent
has to trade off: treat now vs. walk back to restock vs. rest vs. refer.

State changes depend on the actions (position, stock, energy, severity of each household),
so this is a sequential decision problem, not a one-shot prediction.
"""
import gymnasium as gym
import numpy as np
from gymnasium import spaces

# ---- action ids (discrete, 8 actions) ----
NORTH, SOUTH, EAST, WEST, TREAT, REFER, RESTOCK, REST = range(8)
ACTION_NAMES = ["North", "South", "East", "West", "Treat", "Refer", "Restock", "Rest"]
NEED_NAMES = ["Child vaccination", "Malaria treatment", "Antenatal care"]

GRID = 8
N_HOUSES = 6
MAX_STOCK = 6          # doses/kits the HEW can carry
MAX_ENERGY = 40
MAX_STEPS = 80         # one working day: sunrise -> sunset
REFERRAL_SLOTS = 2     # health centre capacity per day
POST = (0, 0)          # health post location


class HealthWorkerEnv(gym.Env):
    metadata = {"render_modes": ["ansi"]}

    def __init__(self, shaping=True):
        super().__init__()
        self.shaping = shaping
        self.action_space = spaces.Discrete(8)
        # 5 agent features + 3 post/referral + 3 visit-list hints + 4 per household
        self.obs_dim = 5 + 3 + 3 + 4 * N_HOUSES
        self.observation_space = spaces.Box(-1.0, 1.0, shape=(self.obs_dim,), dtype=np.float32)
        self.rng = np.random.default_rng()

    # ------------------------------------------------------------------ reset
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        if seed is not None:
            self.rng = np.random.default_rng(seed)
        cells = [(x, y) for x in range(GRID) for y in range(GRID) if (x, y) != POST]
        idx = self.rng.choice(len(cells), N_HOUSES, replace=False)
        self.house_pos = np.array([cells[i] for i in idx])
        self.severity = self.rng.integers(1, 4, N_HOUSES)      # 1 mild, 2 moderate, 3 critical
        self.need = self.rng.integers(0, 3, N_HOUSES)           # which service is needed
        self.pos = np.array(POST)
        self.stock = MAX_STOCK
        self.energy = MAX_ENERGY
        self.t = 0
        self.referrals_left = REFERRAL_SLOTS
        self.referred = np.zeros(N_HOUSES, dtype=bool)
        self.last_action = None
        return self._obs(), {}

    # ------------------------------------------------------------------ helpers
    def _house_at(self):
        for i, p in enumerate(self.house_pos):
            if p[0] == self.pos[0] and p[1] == self.pos[1]:
                return i
        return None

    def _potential(self):
        """Distance-based shaping: go to the nearest sick house if we have medicine, else go home."""
        sick = np.where(self.severity > 0)[0]
        if len(sick) == 0 or self.stock == 0:
            target = np.array(POST)
        else:
            d = np.abs(self.house_pos[sick] - self.pos).sum(axis=1)
            target = self.house_pos[sick[np.argmin(d)]]
        return -float(np.abs(target - self.pos).sum())

    def _obs(self):
        o = [self.pos[0] / (GRID - 1), self.pos[1] / (GRID - 1),
             self.stock / MAX_STOCK, self.energy / MAX_ENERGY, 1 - self.t / MAX_STEPS,
             self.referrals_left / REFERRAL_SLOTS,
             (POST[0] - self.pos[0]) / (GRID - 1), (POST[1] - self.pos[1]) / (GRID - 1)]
        # visit-list hints: severity of the household at this cell + offset to the nearest sick household
        i = self._house_at()
        sick = np.where(self.severity > 0)[0]
        if len(sick):
            d = np.abs(self.house_pos[sick] - self.pos).sum(axis=1)
            nx, ny = self.house_pos[sick[np.argmin(d)]] - self.pos
        else:
            nx = ny = 0
        o += [(self.severity[i] / 3.0) if i is not None else 0.0, nx / (GRID - 1), ny / (GRID - 1)]
        for i in range(N_HOUSES):
            dx, dy = self.house_pos[i] - self.pos
            o += [dx / (GRID - 1), dy / (GRID - 1), self.severity[i] / 3.0, (self.need[i] + 1) / 3.0 * (self.severity[i] > 0)]
        return np.array(o, dtype=np.float32)

    # ------------------------------------------------------------------ step
    def step(self, action):
        action = int(action)
        reward = -0.05                      # small time cost for every step
        terminated = False
        pot_before = self._potential() if self.shaping else 0.0
        i = self._house_at()

        if action <= WEST:                  # movement
            dx, dy = [(0, 1), (0, -1), (1, 0), (-1, 0)][action]
            nx, ny = self.pos[0] + dx, self.pos[1] + dy
            if 0 <= nx < GRID and 0 <= ny < GRID:
                self.pos = np.array([nx, ny])
            else:
                reward -= 0.2               # walked into the boundary
            self.energy -= 1

        elif action == TREAT:
            if i is not None and self.severity[i] > 0 and self.stock > 0:
                reward += 1.0 + 1.0 * self.severity[i]
                self.severity[i] -= 1       # critical (3) -> moderate (2) only; needs more care
                self.stock -= 1
                if self.severity[i] == 0:
                    reward += 2.0           # household fully stable
            else:
                reward -= 1.0               # wasted action / no medicine
            self.energy -= 1

        elif action == REFER:
            if i is not None and self.severity[i] >= 2 and self.referrals_left > 0:
                reward += 4.0 + self.severity[i]
                self.severity[i] = 0
                self.referred[i] = True
                self.referrals_left -= 1
            else:
                reward -= 1.0
            self.energy -= 2

        elif action == RESTOCK:
            if tuple(self.pos) == POST and self.stock < MAX_STOCK:
                reward += 0.3 * (MAX_STOCK - self.stock)
                self.stock = MAX_STOCK
            else:
                reward -= 1.0
            self.energy -= 1

        elif action == REST:
            self.energy = min(MAX_ENERGY, self.energy + 4)

        # disease progression: untreated households may get worse (delayed consequence)
        for j in range(N_HOUSES):
            if 0 < self.severity[j] < 3 and self.rng.random() < 0.02:
                self.severity[j] += 1
                if self.severity[j] == 3:
                    reward -= 2.0           # household became critical

        if self.shaping:
            reward += 0.2 * (self._potential() - pot_before)

        self.t += 1
        self.last_action = action

        # ---- terminal conditions ----
        if np.all(self.severity == 0):
            reward += 10.0                  # every household stable
            terminated = True
        elif self.energy <= 0:
            reward -= 10.0                  # exhausted HEW
            terminated = True
        truncated = (self.t >= MAX_STEPS) and not terminated   # sunset
        if truncated:
            reward -= 5.0 * int(np.sum(self.severity == 3))      # critical patients left at sunset
        return self._obs(), float(reward), terminated, truncated, {"severity": self.severity.copy()}

    # ------------------------------------------------------------------ misc
    def state_dict(self):
        """Full state as plain python (used for the three.js replay)."""
        return {
            "pos": [int(self.pos[0]), int(self.pos[1])],
            "stock": int(self.stock), "energy": int(self.energy), "t": int(self.t),
            "ref": int(self.referrals_left),
            "sev": [int(s) for s in self.severity],
            "action": self.last_action,
        }

    def layout(self):
        return {"houses": [[int(x), int(y)] for x, y in self.house_pos],
                "need": [int(n) for n in self.need], "post": list(POST)}

    def render(self):
        g = [["." for _ in range(GRID)] for _ in range(GRID)]
        g[POST[1]][POST[0]] = "P"
        for k, (x, y) in enumerate(self.house_pos):
            g[y][x] = str(self.severity[k])
        g[self.pos[1]][self.pos[0]] = "H" if g[self.pos[1]][self.pos[0]] == "." else "*"
        return "\n".join("".join(r) for r in g[::-1])
