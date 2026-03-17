from typing import SupportsFloat, Any
from gymnasium import Env, spaces
import numpy as np
from gymnasium.core import ActType, ObsType, RenderFrame
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.callbacks import CheckpointCallback

from ensemble import Ensemble


class EnsembleWeightEnv(Env):
    def __init__(self, ensemble_model: Ensemble):
        super(EnsembleWeightEnv, self).__init__()
        self.ensemble_model = ensemble_model

        # initial weights
        self.weights = np.repeat(1.0, len(self.ensemble_model.models))
        self.weights = self.weights / np.sum(self.weights)

        # Action space
        self.action_space = spaces.Box(low=0, high=1, shape=(len(self.ensemble_model.models),), dtype=np.float32)
        # Observation space
        self.observation_space = spaces.Box(low=0, high=1, shape=(1,), dtype=np.float32)

    def reset(self, **kwargs):
        # Reset the environment:
        self.weights = np.repeat(1.0, len(self.ensemble_model.models))
        self.weights = self.weights / np.sum(self.weights)
        return np.array(self.weights)

    def step(
        self, action: ActType
    ) -> tuple[ObsType, SupportsFloat, bool, bool, dict[str, Any]]:
        self.weights = np.array(action)
        self.weights = self.weights / np.sum(self.weights)

        reward = self.ensemble_model.test(self.weights.tolist())
        obs = np.array(self.weights)
        return obs, reward, False, False, {}

    def render(self) -> RenderFrame | list[RenderFrame] | None:
        pass


def train(ensemble_model: Ensemble, n_steps: int = 2048, n_epochs: int = 10,batch_size: int = 64, device: str = 'cuda',
          save_path: str = 'staric_rl_model_ppo', total_timesteps: int = 10240 * 100):
    env = DummyVecEnv([lambda: EnsembleWeightEnv(ensemble_model)])
    rl_model = PPO('MlpPolicy',
                   env,
                   n_steps=n_steps,
                   n_epochs=n_epochs,
                   batch_size=batch_size,
                   verbose=1,
                   device=device)
    checkpoint_callback = CheckpointCallback(
        save_freq=n_steps,
        save_path=save_path,
        name_prefix='ppo_static',
        verbose=1
    )
    rl_model.learn(total_timesteps=total_timesteps, callback=checkpoint_callback)

    # After training, reset the environment to get the initial observation
    obs = env.reset()

    # Predict the weights with the trained policy
    final_weights, _ = rl_model.predict(obs)

    # Print the final weights
    print("Final weights found by PPO:", final_weights)
