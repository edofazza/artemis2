import torch
from gymnasium import Env, spaces
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.callbacks import CheckpointCallback
from torch import nn
import os
from torchmetrics.classification import MultilabelAveragePrecision

from utils.utils import AverageMeter


class EnsembleWeightEnv(Env):
    def __init__(self, models, dataloader, device):
        super(EnsembleWeightEnv, self).__init__()
        self.dataloader = dataloader
        self.models = models
        self.device = device
        self.dataloader_iter = iter(dataloader)
        #self.criterion = nn.BCEWithLogitsLoss()
        self.criterion = MultilabelAveragePrecision(num_labels=140, average='micro')

        # Observation space is a tensor of shape (16 * 3, 244, 244)
        self.observation_shape = (16*3, 224, 224)
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=self.observation_shape, dtype=np.float32
        )

        # Action space: weights for ensemble of `n_models`
        self.action_space = spaces.Box(low=0.0, high=1.0, shape=(len(self.models),), dtype=np.float32)

        # Current observation
        self.current_observation = None
        self.current_description = None
        self.current_target = None

    def reset(self, **kwargs):
        # reset the dataloader
        self.dataloader_iter = iter(self.dataloader)
        # return the first element
        self.current_observation, self.current_description, self.current_target = next(self.dataloader_iter)
        b, f, c, h, w = self.current_observation.numpy().shape
        return np.reshape(self.current_observation.numpy(), newshape=(b*f*c, h, w)), {}

    def step(self, action):
        for model in self.models:
            model.eval()

        action_sum = sum(action)
        if action_sum != 0:
            weights = [w / action_sum for w in action]
        else:
            weights = [0 for _ in action]
            # take action on the observation
        with torch.no_grad():
            self.current_observation, self.current_description, self.current_target = (self.current_observation.to(self.device),
                                                                                       self.current_description.to(self.device),
                                                                                       self.current_target.long().to(self.device))
            tmp_results = self.models[0](self.current_observation, self.current_description)
            if type(tmp_results) is tuple:
                tmp_results = tmp_results[0]
            weighted_predictions = tmp_results * weights[0]
            for i in range(1, len(self.models)):
                tmp_results = self.models[i](self.current_observation, self.current_description)
                if type(tmp_results) is tuple:
                    tmp_results = tmp_results[0]
                weighted_predictions += tmp_results * weights[i]
        # evaluate loss on observation -> reward (closer to zero the better, -loss)
            reward = self.criterion(weighted_predictions, self.current_target)
        # update the new observation
        # if not more observations then reset
        try:
            self.current_observation, self.current_description, self.current_target = next(self.dataloader_iter)
        except StopIteration:
            self.reset()
        b, f, c, h, w = self.current_observation.numpy().shape
        return np.reshape(self.current_observation.numpy(), newshape=(b*f*c, h, w)), reward, False, False, {}

    def render(self, **kwargs):
        return None, {}


def train(models, dataloader, device, fold):
    if not os.path.exists('rl'):
        os.mkdir('rl')
    # Number of steps should be a multiple of the elements inside dataloader
    # update n_steps should be equal to elements inside dataloader
    env = EnsembleWeightEnv(models, dataloader, device)
    env = DummyVecEnv([lambda: env])

    # Initialize the PPO model
    model = PPO('CnnPolicy',
                env,
                verbose=1,
                n_steps=len(dataloader),
                batch_size=64,
                n_epochs=3,
                device=device,
                policy_kwargs=dict(normalize_images=False),
                )
    checkpoint_callback = CheckpointCallback(
        save_freq=len(dataloader)*15,
        save_path=f'rl',
        name_prefix=f'ppo_fold{fold}_2',
        verbose=1
    )

    # Train the PPO model
    model.learn(total_timesteps=len(dataloader) * 250, callback=checkpoint_callback) # 150


def test(models, dataloader, device):
    for model in models:
        model.eval()
    eval_meter = AverageMeter()
    eval_metrics = MultilabelAveragePrecision(num_labels=140, average='micro')
    total_batches = len(dataloader)
    batch_idx = 0
    for data, summaries, label in dataloader:
        print('Batch {}/{}'.format(batch_idx, total_batches), flush=True)
        batch_idx += 1
        data, summaries, label = (data.to(device),
                                  summaries.to(device),
                                  label.long().to(device))
        b, f, c, h, w = data.shape
        model_ppo = PPO.load('rl/ppo_fold2_2_1152000_steps.zip', device=device)
        with torch.no_grad():
            weights, _ = model_ppo.predict(np.reshape(data.cpu().numpy(), newshape=(b*f*c, h, w)))
            action_sum = sum(weights)
            if action_sum != 0:
                weights = [w / action_sum for w in weights]
            else:
                weights = [0 for _ in weights]

            tmp_results = models[0](data, summaries)
            if type(tmp_results) is tuple:
                tmp_results = tmp_results[0]
            weighted_predictions = tmp_results * weights[0]
            for i in range(1, len(models)):
                tmp_results = models[i](data, summaries)
                if type(tmp_results) is tuple:
                    tmp_results = tmp_results[0]
                weighted_predictions += tmp_results * weights[i]
            eval_this = eval_metrics(weighted_predictions, label)
            eval_meter.update(eval_this.item(), data.shape[0])
    print("[INFO] Evaluation Metric: {:.2f}".format(eval_meter.avg * 100), flush=True)
