# Padlock Gymnasium Env

Padlock Gymnasium Env is a simple gymnasium environment that models a padlock with 4 discs, each with 4 symbols. The agent's goal is to find the secret combination of symbols to unlock the padlock.

The action space is a MultiDiscrete space with 2 dimensions, representing the selected disc and symbol. The observation space is a Box space of shape (4, 2) representing the current combination and a valid mask, where the valid mask is a binary array of shape (4,), with 1 indicating that the corresponding symbol in the current combination is correct and 0 indicating otherwise.

## Install it from PyPI

```bash
pip install padlock_env
```

## Install it locally

```bash
pip install -e .
```

## Usage

Here's an example of how to use the environment:

```python
import gymnasium
import padlock_env

env = gymnasium.make("PadlockEnv-v0")
observations, _ = env.reset()

done = False
while not done:
    action = env.action_space.sample()
    observations, reward, done, _, _ = env.step(action)
    print(f"Observations: {observations}, Reward: {reward}, Done: {done}")
```

## Development

Read the [CONTRIBUTING.md](CONTRIBUTING.md) file.
