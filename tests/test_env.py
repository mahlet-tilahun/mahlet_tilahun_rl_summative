import numpy as np
from environment import HealthWorkerEnv


def test_reset_and_obs_shape():
    env = HealthWorkerEnv()
    obs, _ = env.reset(seed=1)
    assert obs.shape == (env.obs_dim,) and obs.dtype == np.float32


def test_random_episode_terminates():
    env = HealthWorkerEnv()
    env.reset(seed=2)
    for _ in range(200):
        _, _, term, trunc, _ = env.step(env.action_space.sample())
        if term or trunc:
            return
    raise AssertionError("episode never ended")


def test_restock_only_at_post():
    env = HealthWorkerEnv()
    env.reset(seed=3)
    env.stock = 0
    _, r, *_ = env.step(6)   # restock at the post
    assert env.stock == 6 and r > 0
