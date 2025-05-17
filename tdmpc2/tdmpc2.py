import torch
import torch.nn.functional as F

from .common import math
from .common.scale import RunningScale
from .common.world_model import WorldModel
from .common.layers import api_model_conversion
from tensordict import TensorDict


class TDMPC2(torch.nn.Module):
	"""
	TD-MPC2 agent. Implements training + inference.
	Can be used for both single-task and multi-task experiments,
	and supports both state and pixel observations.
	"""

	def __init__(self, cfg, device: str = 'cuda:0'):
		super().__init__()

		capturable = torch.cuda.is_available()

		self.cfg = cfg
		self.device = torch.device(device)
		self.model = WorldModel(cfg).to(self.device)
		self.optim = torch.optim.Adam([
			{'params': self.model._encoder.parameters(), 'lr': self.cfg.lr*self.cfg.enc_lr_scale},
			{'params': self.model._dynamics.parameters()},
			{'params': self.model._reward.parameters()},
			{'params': self.model._termination.parameters() if self.cfg.episodic else []},
			{'params': self.model._Qs.parameters()},
			{'params': self.model._task_emb.parameters() if self.cfg.multitask else []
			 }
		], lr=self.cfg.lr, capturable=capturable)
		self.pi_optim = torch.optim.Adam(self.model._pi.parameters(), lr=self.cfg.lr, eps=1e-5, capturable=capturable)
		self.model.eval()
		self.scale = RunningScale(cfg, device=device)
		self.cfg.iterations += 2*int(cfg.action_dim >= 20) # Heuristic for large action spaces
		self.discount = torch.tensor(
			[self._get_discount(ep_len) for ep_len in cfg.episode_lengths], device=device
		) if self.cfg.multitask else self._get_discount(cfg.episode_length)
		print('Episode length:', cfg.episode_length)
		print('Discount factor:', self.discount)
		self._prev_mean = torch.nn.Buffer(torch.zeros(self.cfg.horizon, self.cfg.action_dim, device=self.device))
		if cfg.compile:
			print('Compiling update function with torch.compile...')
			self._update = torch.compile(self._update, mode="reduce-overhead")

	@property
	def plan(self):
		_plan_val = getattr(self, "_plan_val", None)
		if _plan_val is not None:
			return _plan_val
		if self.cfg.compile:
			plan = torch.compile(self._plan, mode="reduce-overhead")
		else:
			plan = self._plan
		self._plan_val = plan
		return self._plan_val

	def _get_discount(self, episode_length):
		"""
		Returns discount factor for a given episode length.
		Simple heuristic that scales discount linearly with episode length.
		Default values should work well for most tasks, but can be changed as needed.

		Args:
			episode_length (int): Length of the episode. Assumes episodes are of fixed length.

		Returns:
			float: Discount factor for the task.
		"""
		frac = episode_length/self.cfg.discount_denom
		return min(max((frac-1)/(frac), self.cfg.discount_min), self.cfg.discount_max)

	def save(self, fp):
		"""
		Save state dict of the agent to filepath.

		Args:
			fp (str): Filepath to save state dict to.
		"""
		torch.save({"model": self.model.state_dict()}, fp)

	def load(self, fp):
		"""
		Load a saved state dict from filepath (or dictionary) into current agent.

		Args:
			fp (str or dict): Filepath or state dict to load.
		"""
		if isinstance(fp, dict):
			state_dict = fp
		else:
			state_dict = torch.load(fp, map_location=torch.get_default_device(), weights_only=False)
		state_dict = state_dict["model"] if "model" in state_dict else state_dict
		state_dict = api_model_conversion(self.model.state_dict(), state_dict)
		self.model.load_state_dict(state_dict)
		return

	@torch.no_grad()
	def act(self, obs, t0=False, eval_mode=False, task=None):
		"""
		Select an action by planning in the latent space of the world model.

		Args:
			obs (torch.Tensor): Observation from the environment.
			t0 (bool): Whether this is the first observation in the episode.
			eval_mode (bool): Whether to use the mean of the action distribution.
			task (int): Task index (only used for multi-task experiments).

		Returns:
			torch.Tensor: Action to take in the environment.
		"""
		obs = obs.to(self.device, non_blocking=True).unsqueeze(0)
		if task is not None:
			task = torch.tensor([task], device=self.device)
		if self.cfg.mpc:
			return self.plan(obs, t0=t0, eval_mode=eval_mode, task=task).cpu()

		z = self.model.encode(obs, task)
		action, info = self.model.pi(z, task)
		if eval_mode:
			action = info["mean"]

		action_index = action[0].cpu()

		assert action_index.shape == (1,), f"Unexpected action_index shape: {action_index.shape}"

		return action_index

	@torch.no_grad()
	def _estimate_value(self, z, actions, task):
		"""Estimate value of a trajectory starting at latent state z and executing given actions."""
		G, discount = 0, 1
		termination = torch.zeros(self.cfg.num_samples, 1, dtype=torch.float32, device=z.device)
		for t in range(self.cfg.horizon):
			reward = math.two_hot_inv(self.model.reward(z, actions[t], task), self.cfg)
			# print(f"_plan actions[t]: {actions[t].shape}")
			z = self.model.next(z, actions[t], task)
			G = G + discount * (1-termination) * reward
			discount_update = self.discount[torch.tensor(task)] if self.cfg.multitask else self.discount
			discount = discount * discount_update
			if self.cfg.episodic:
				termination = torch.clip(termination + (self.model.termination(z, task) > 0.5).float(), max=1.)
		action, _ = self.model.pi(z, task)
		return G + discount * (1-termination) * self.model.Q(z, action, task, return_type='avg')

	@torch.no_grad()
	def _plan(self, obs, t0=False, eval_mode=False, task=None):
		"""
		Plan a sequence of actions using the learned world model.

		Args:
			z (torch.Tensor): Latent state from which to plan.
			t0 (bool): Whether this is the first observation in the episode.
			eval_mode (bool): Whether to use the mean of the action distribution.
			task (Torch.Tensor): Task index (only used for multi-task experiments).

		Returns:
			torch.Tensor: Action to take in the environment.
		"""
		# Sample policy trajectories
		z = self.model.encode(obs, task)
		if self.cfg.num_pi_trajs > 0:
			# Sample policy-guided actions (index-based)
			pi_actions = torch.empty(self.cfg.horizon, self.cfg.num_pi_trajs, dtype=torch.long, device=self.device)
			_z = z.repeat(self.cfg.num_pi_trajs, 1)
			for t in range(self.cfg.horizon - 1):
				logits, _ = self.model.pi(_z, task)
				pi_actions[t] = logits.argmax(dim=-1)

				# print(f"_plan pi_actions[t]: {pi_actions[t].shape}")
				_z = self.model.next(_z, pi_actions[t], task)
			logits, _ = self.model.pi(_z, task)
			pi_actions[-1] = logits.argmax(dim=-1)

		# Initialize state and parameters
		z = z.repeat(self.cfg.num_samples, 1)

		# actions = torch.empty(self.cfg.horizon, self.cfg.num_samples, self.cfg.action_dim, device=self.device)
		actions = torch.empty(self.cfg.horizon, self.cfg.num_samples, dtype=torch.long, device=self.device)

		if self.cfg.num_pi_trajs > 0:
			actions[:, :self.cfg.num_pi_trajs] = pi_actions

		if _USE_CONTINUOUS_ACTIONS := False:
			mean = torch.zeros(self.cfg.horizon, self.cfg.action_dim, device=self.device)
			std = torch.full((self.cfg.horizon, self.cfg.action_dim), self.cfg.max_std, dtype=torch.float, device=self.device)
			if not t0:
				mean[:-1] = self._prev_mean[1:]

			# Iterate MPPI
			for _ in range(self.cfg.iterations):
				# Sample actions
				r = torch.randn(self.cfg.horizon, self.cfg.num_samples-self.cfg.num_pi_trajs, self.cfg.action_dim, device=std.device)
				actions_sample = mean.unsqueeze(1) + std.unsqueeze(1) * r
				actions_sample = actions_sample.clamp(-1, 1)
				actions[:, self.cfg.num_pi_trajs:] = actions_sample
				if self.cfg.multitask:
					actions = actions * self.model._action_masks[task]

				# Compute elite actions
				value = self._estimate_value(z, actions, task).nan_to_num(0)
				elite_idxs = torch.topk(value.squeeze(1), self.cfg.num_elites, dim=0).indices
				elite_value, elite_actions = value[elite_idxs], actions[:, elite_idxs]

				# Update parameters
				max_value = elite_value.max(0).values
				score = torch.exp(self.cfg.temperature*(elite_value - max_value))
				score = score / score.sum(0)
				mean = (score.unsqueeze(0) * elite_actions).sum(dim=1) / (score.sum(0) + 1e-9)
				std = ((score.unsqueeze(0) * (elite_actions - mean.unsqueeze(1)) ** 2).sum(dim=1) / (score.sum(0) + 1e-9)).sqrt()
				std = std.clamp(self.cfg.min_std, self.cfg.max_std)
				if self.cfg.multitask:
					mean = mean * self.model._action_masks[task]
					std = std * self.model._action_masks[task]

			# Select action
			rand_idx = math.gumbel_softmax_sample(score.squeeze(1))
			actions = torch.index_select(elite_actions, 1, rand_idx).squeeze(1)
			a, std = actions[0], std[0]
			if not eval_mode:
				a = a + std * torch.randn(self.cfg.action_dim, device=std.device)
			self._prev_mean.copy_(mean)
			return a.clamp(-1, 1)

		elif _USE_ONEHOT := False:
			# MPPI Loop
			for _ in range(self.cfg.iterations):
				# Sample discrete actions
				actions_sample = torch.randint(0, self.cfg.action_dim, (self.cfg.horizon, self.cfg.num_samples - self.cfg.num_pi_trajs), device=self.device)
				actions_onehot = F.one_hot(actions_sample, num_classes=self.cfg.action_dim).float()
				actions[:, self.cfg.num_pi_trajs:] = actions_onehot

				# Evaluate and select elites
				value = self._estimate_value(z, actions, task).nan_to_num(0)
				elite_idxs = torch.topk(value.squeeze(1), self.cfg.num_elites, dim=0).indices
				elite_value, elite_actions = value[elite_idxs], actions[:, elite_idxs]

				# Majority vote for each timestep
				elite_indices = elite_actions.argmax(dim=-1)  # (horizon, num_elites)
				mode_actions = torch.mode(elite_indices, dim=1).values  # (horizon,)
				actions = F.one_hot(mode_actions, num_classes=self.cfg.action_dim).float().unsqueeze(1).repeat(1, self.cfg.num_samples, 1)

			# Select action
			if False:
				first_action = mode_actions[0]
				return F.one_hot(first_action, num_classes=self.cfg.action_dim).float()
			else:
				sample_probs = elite_value.softmax(0)  # or use score
				idx = torch.multinomial(sample_probs, 1)
				return elite_actions[0, idx].squeeze(0)

		else:
			# MPPI Loop
			# Assume:
			#   self.cfg.horizon = H
			#   self.cfg.num_samples = N
			#   self.cfg.num_pi_trajs = P
			#   self.cfg.num_elites = E
			#   self.cfg.action_dim = A
			#   z.shape = (N, latent_dim)
			#   actions: preallocated as shape (H, N), dtype=torch.long

			for _ in range(self.cfg.iterations):
				# Sample discrete actions for remaining trajectories
				actions_sample = torch.randint(
					0, self.cfg.num_discrete_actions,
					(self.cfg.horizon, self.cfg.num_samples - self.cfg.num_pi_trajs),  # shape: (H, N - P)
					device=self.device
				)
				# DEBUG PRINT: (H, N - P)
				# print(f"ACTIONS SAMPLE, high={self.cfg.num_discrete_actions} shape: {actions_sample.shape} actions shape: {actions.shape} {actions_sample=}")

				# Fill sampled actions into the remaining slots
				actions[:, self.cfg.num_pi_trajs:] = actions_sample  # actions shape: (H, N)

				# Compute estimated value for each action sequence
				value = self._estimate_value(z, actions, task).nan_to_num(0)  # shape: (N, 1)

				# Select top-k elite action sequences by value
				elite_idxs = torch.topk(value.squeeze(1), self.cfg.num_elites, dim=0).indices  # shape: (E,)
				elite_value = value[elite_idxs]                    # shape: (E, 1)
				elite_actions = actions[:, elite_idxs]             # shape: (H, E)

				# Majority vote across elites per timestep
				#
				# NOTE: elite_actions moved to cpu() because torch.mode() isn't available on device 'mps'.
				mode_actions = torch.mode(elite_actions.cpu(), dim=1).values  # shape: (H,)

				# Repeat mode_actions across all samples, in-place
				# mode_actions[:, None] shape: (H, 1) → broadcast to (H, N) in assignment
				actions[:] = mode_actions[:, None]  # actions shape: (H, N) after write

			# Return first action (discrete index)
			if eval_mode:
				return mode_actions[0]  # (int tensor)
			else:
				probs = elite_value.softmax(0)  # (num_elites,)
				# print(f"PROBS shape: {probs.shape}")
				idx = torch.multinomial(probs.squeeze(1), 1).item()
				return elite_actions[0, idx]    # (int tensor)


	def update_pi(self, zs, task):
		"""
		Update policy using a sequence of latent states.

		Args:
			zs (torch.Tensor): Sequence of latent states.
			task (torch.Tensor): Task index (only used for multi-task experiments).

		Returns:
			float: Loss of the policy update.
		"""
		action, info = self.model.pi(zs, task)
		qs = self.model.Q(zs, action, task, return_type='avg', detach=True)
		self.scale.update(qs[0])
		qs = self.scale(qs)

		# Loss is a weighted sum of Q-values
		rho = torch.pow(self.cfg.rho, torch.arange(len(qs), device=self.device))
		pi_loss = (-(self.cfg.entropy_coef * info["scaled_entropy"] + qs).mean(dim=(1,2)) * rho).mean()
		pi_loss.backward()
		pi_grad_norm = torch.nn.utils.clip_grad_norm_(self.model._pi.parameters(), self.cfg.grad_clip_norm)
		self.pi_optim.step()
		self.pi_optim.zero_grad(set_to_none=True)

		info = TensorDict({
			"pi_loss": pi_loss,
			"pi_grad_norm": pi_grad_norm,
			"pi_entropy": info["entropy"],
			"pi_scaled_entropy": info["scaled_entropy"],
			"pi_scale": self.scale.value,
		})
		return info

	@torch.no_grad()
	def _td_target(self, next_z, reward, terminated, task):
		"""
		Compute the TD-target from a reward and the observation at the following time step.

		Args:
			next_z (torch.Tensor): Latent state at the following time step.
			reward (torch.Tensor): Reward at the current time step.
			terminated (torch.Tensor): Termination signal at the current time step.
			task (torch.Tensor): Task index (only used for multi-task experiments).

		Returns:
			torch.Tensor: TD-target.
		"""
		action, _ = self.model.pi(next_z, task)
		discount = self.discount[task].unsqueeze(-1) if self.cfg.multitask else self.discount
		return reward + discount * (1-terminated) * self.model.Q(next_z, action, task, return_type='min', target=True)

	def _update(self, obs, action, reward, terminated, task=None):
		# Compute targets
		with torch.no_grad():
			next_z = self.model.encode(obs[1:], task)
			td_targets = self._td_target(next_z, reward, terminated, task)

		# Prepare for update
		self.model.train()

		# Latent rollout
		zs = torch.empty(self.cfg.horizon+1, self.cfg.batch_size, self.cfg.latent_dim, device=self.device)
		z = self.model.encode(obs[0], task)
		zs[0] = z
		consistency_loss = 0
		for t, (_action, _next_z) in enumerate(zip(action.unbind(0), next_z.unbind(0))):
			# print(f"_plan _update[t]: {_action.shape}")
			z = self.model.next(z, _action, task)
			consistency_loss = consistency_loss + F.mse_loss(z, _next_z) * self.cfg.rho**t
			zs[t+1] = z

		# print(f"Q _update[t]: {_action.shape}")
		b_action = action.unsqueeze(-1)

		# Predictions
		_zs = zs[:-1]
		qs = self.model.Q(_zs, b_action, task, return_type='all')
		reward_preds = self.model.reward(_zs, action, task)
		if self.cfg.episodic:
			termination_pred = self.model.termination(zs[1:], task, unnormalized=True)

		# Compute losses
		reward_loss, value_loss = 0, 0
		for t, (rew_pred_unbind, rew_unbind, td_targets_unbind, qs_unbind) in enumerate(zip(reward_preds.unbind(0), reward.unbind(0), td_targets.unbind(0), qs.unbind(1))):
			reward_loss = reward_loss + math.soft_ce(rew_pred_unbind, rew_unbind, self.cfg).mean() * self.cfg.rho**t
			for _, qs_unbind_unbind in enumerate(qs_unbind.unbind(0)):
				value_loss = value_loss + math.soft_ce(qs_unbind_unbind, td_targets_unbind, self.cfg).mean() * self.cfg.rho**t

		consistency_loss = consistency_loss / self.cfg.horizon
		reward_loss = reward_loss / self.cfg.horizon
		if self.cfg.episodic:
			termination_loss = F.binary_cross_entropy_with_logits(termination_pred, terminated)
		else:
			termination_loss = 0.
		value_loss = value_loss / (self.cfg.horizon * self.cfg.num_q)
		total_loss = (
			self.cfg.consistency_coef * consistency_loss +
			self.cfg.reward_coef * reward_loss +
			self.cfg.termination_coef * termination_loss +
			self.cfg.value_coef * value_loss
		)

		# Update model
		total_loss.backward()
		grad_norm = torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.cfg.grad_clip_norm)
		self.optim.step()
		self.optim.zero_grad(set_to_none=True)

		# Update policy
		pi_info = self.update_pi(zs.detach(), task)

		# Update target Q-functions
		self.model.soft_update_target_Q()

		# Return training statistics
		self.model.eval()
		info = TensorDict({
			"consistency_loss": consistency_loss,
			"reward_loss": reward_loss,
			"value_loss": value_loss,
			"termination_loss": termination_loss,
			"total_loss": total_loss,
			"grad_norm": grad_norm,
		})
		if self.cfg.episodic:
			info.update(math.termination_statistics(torch.sigmoid(termination_pred[-1]), terminated[-1]))
		info.update(pi_info)
		return info.detach().mean()

	def update(self, buffer):
		"""
		Main update function. Corresponds to one iteration of model learning.

		Args:
			buffer (common.buffer.Buffer): Replay buffer.

		Returns:
			dict: Dictionary of training statistics.
		"""
		obs, action, reward, terminated, task = buffer.sample()

		# print(f"ACTION SHAPE FROM BUFFER: {action.shape}")

		# action = action.unsqueeze(-1)

		kwargs = {}
		if task is not None:
			kwargs["task"] = task
		torch.compiler.cudagraph_mark_step_begin()

		# print(f"SHAPE PASSED IN TO _update: {action.shape}")

		return self._update(obs, action, reward, terminated, **kwargs)
