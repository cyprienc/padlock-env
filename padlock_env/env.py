from typing import Any, Optional, Tuple

import gymnasium
import numpy as np
from gymnasium import spaces
from gymnasium.utils import seeding
from numpy.typing import NDArray


class PadlockEnv(gymnasium.Env):
    """Padlock environment for Gymnasium.

    This environment models a padlock with 4 discs, each with 4 symbols.

    The agent's goal is to find the secret combination of
    symbols to unlock the padlock.

    The action space is a MultiDiscrete space with 2 dimensions,
    representing the selected disc and symbol.

    The observation space is a Box space of shape (4, 2),
    representing the current combination and a valid mask,
    where the valid mask is a binary array of shape (4,),
    with 1 indicating that the corresponding
    symbol in the combination is correct
    and 0 indicating otherwise.
    """

    N_DISCS: int = 4
    N_SYMBOLS: int = 4

    action_space = spaces.MultiDiscrete([N_DISCS, N_SYMBOLS])
    observation_space = spaces.Box(
        low=0,
        high=np.asarray([[N_SYMBOLS, 1]] * N_DISCS),
        shape=(N_DISCS, 2),
        dtype=int,
    )

    def __init__(self, seed: Optional[int] = None):
        """Initialize the padlock environment.
        Args:
            seed (int): The seed to initialize the random number generator.
        """
        super().__init__()
        self.np_random, _ = seeding.np_random(seed)

        self._secret_combination: Optional[NDArray] = None
        self._combination: Optional[NDArray] = None

    def reset(
        self,
        *,
        seed: Optional[int] = None,
        options: Optional[dict] = None,
    ) -> Tuple[NDArray, dict]:
        """Reset the environment and return the initial observation.
        Args:
            seed (int): The seed to initialize the random number generator.
            options (dict): Additional options for the environment.
        Returns:
            Tuple[NDArray, dict]:
                The initial observation and an empty info dict.
        """
        super().reset(seed=seed, options=options)

        self._secret_combination = self.np_random.integers(
            low=0,
            high=self.N_SYMBOLS,
            size=self.N_DISCS,
            dtype=self.observation_space.dtype,
        )

        combination = self.np_random.integers(
            low=0,
            high=self.N_SYMBOLS,
            size=self.N_DISCS,
            dtype=self.observation_space.dtype,
        )

        while np.alltrue(self._secret_combination == combination):
            combination = self.np_random.integers(
                low=0,
                high=self.N_SYMBOLS,
                size=self.N_DISCS,
                dtype=self.observation_space.dtype,
            )

        self._combination = combination
        valid_mask = self._compute_valid_mask(combination).astype(
            self.observation_space.dtype
        )
        observations = np.hstack(
            (self._combination.reshape(-1, 1), valid_mask.reshape(-1, 1))
        )
        return observations, {}

    def step(self, action: NDArray) -> Tuple[NDArray, float, bool, bool, dict]:
        """Step the environment by taking an action.
        Args:
            action (NDArray): The action to take.
        Returns:
            Tuple[NDArray, float, bool, bool, dict]:
                The new observation,
                reward,
                done flag,
                truncated flag
                and an empty info dict.
        """
        assert self._combination is not None
        assert self._secret_combination is not None

        disk, symbol = action
        disk = int(disk)
        symbol = int(symbol)

        assert 0 <= disk < self.N_DISCS
        assert 0 <= symbol < self.N_SYMBOLS

        self._combination[disk] = symbol
        valid_mask = self._compute_valid_mask(self._combination).astype(
            self.observation_space.dtype
        )
        observations = np.hstack(
            (self._combination.reshape(-1, 1), valid_mask.reshape(-1, 1))
        )
        reward = -1.0
        terminated = bool(np.all(valid_mask))
        truncated = False
        info: dict[str, Any] = {}

        return observations, reward, terminated, truncated, info

    def render(self) -> None:
        raise NotImplementedError()

    def _compute_valid_mask(self, combination: NDArray) -> NDArray:
        """Compute the valid mask of the current combination.
        Args:
            combination (NDArray): The current combination.
        Returns:
            NDArray: The valid mask of the current combination.
        """
        return combination == self._secret_combination
