import gymnasium
import numpy as np

from padlock_env.env import PadlockEnv


def test_reset__does_not_smoke(env: PadlockEnv) -> None:
    obs, extra = env.reset()
    assert obs.shape == PadlockEnv.observation_space.shape
    assert obs.dtype == PadlockEnv.observation_space.dtype
    assert np.all(obs >= 0)
    assert np.all(obs[:, 0] < env.N_SYMBOLS)
    assert np.all(obs[:, 1] < 2)


def test_compute_valid_mask__values_are_correct(env: PadlockEnv) -> None:
    combination = np.asarray([0, 1, 2, 3])
    secret_combination = np.asarray([1, 1, 2, 3])
    expected_valid_mask = np.asarray([False, True, True, True])
    env._secret_combination = secret_combination

    actual_valid_mask = env._compute_valid_mask(combination)

    np.testing.assert_equal(actual_valid_mask, expected_valid_mask)


def test_step__final_step(env: PadlockEnv) -> None:
    secret_combination = np.asarray([0, 1, 2, 3])
    combination = np.asarray([1, 1, 2, 3])
    env._combination = combination
    env._secret_combination = secret_combination

    action = np.asarray([0, 0])

    observations, reward, terminated, truncated, info = env.step(action)

    assert observations.shape == env.observation_space.shape
    assert observations.dtype == env.observation_space.dtype

    np.testing.assert_equal(observations[:, 0], secret_combination)
    np.testing.assert_equal(observations[:, 1], np.ones_like(secret_combination))
    assert reward == -1.0
    assert terminated
    assert not truncated


def test_make_env__do_one_step() -> None:
    env = gymnasium.make("Padlock-v0", seed=0)
    env.action_space.seed(0)

    _ = env.reset()
    action = env.action_space.sample()
    _ = env.step(action)
