from collections import defaultdict

import gymnasium as gym
import numpy as np
import torch
import torch.nn.functional as F
from typing import Optional


class TensorWrapper(gym.Wrapper):
	"""
	Wrapper for converting numpy arrays to torch tensors.
	"""

	def __init__(self, env):
		super().__init__(env)

	def rand_act(self):
		# print(f"RAND ACT, ACTION SPACE: {self.action_space=} sample={self.action_space.sample()}")
		action_index = self.action_space.sample()
		return action_index

		#action_onehot = F.one_hot(torch.tensor(action_index), num_classes=self.action_space.n).float()
		#return action_onehot

		# return torch.from_numpy(self.action_space.sample().astype(np.float32))

	def _try_f32_tensor(self, x):
		if isinstance(x, np.ndarray):
			x = torch.from_numpy(x)
			if x.dtype == torch.float64:
				x = x.float()
		return x

	def _obs_to_tensor(self, obs):
		# TODO(millman): what shape is this supposed to be?
		return obs

		if isinstance(obs, dict):
			for k in obs.keys():
				obs[k] = self._try_f32_tensor(obs[k])
		else:
			obs = self._try_f32_tensor(obs)
		return obs

	def reset(self, task=None, seed: Optional[int] = None, options: Optional[dict] = None):
		obs, info = self.env.reset(seed=seed, options=options)
		return self._obs_to_tensor(obs), info

	def step(self, action):
		if isinstance(action, torch.Tensor):
			action = action.numpy()

		obs, reward, done, truncated, info = self.env.step(action)
		info = defaultdict(float, info)

		# print(f"RECEIVED INFO FROM STEP: {info}")

		info['success'] = float(info['success'])
		info['terminated'] = torch.tensor(done)
		return self._obs_to_tensor(obs), torch.tensor(reward, dtype=torch.float32), done, truncated, info
