import torch
import torch.nn.functional as F

from .common import math
from .common.scale import RunningScale
from .common.world_model import WorldModel
from .common.layers import api_model_conversion
from tensordict import TensorDict
from torch.distributions.categorical import Categorical


def symlog(x):
	"""
	Symmetric logarithmic function.
	Adapted from https://github.com/danijar/dreamerv3.
	"""
	return torch.sign(x) * torch.log(1 + torch.abs(x))

def two_hot_debug(x, cfg):
	"""Converts a batch of scalars to soft two-hot encoded targets for discrete regression."""
	if cfg.num_bins == 0:
		return x
	elif cfg.num_bins == 1:
		return symlog(x)
	x = torch.clamp(symlog(x), cfg.vmin, cfg.vmax).squeeze(1)
	bin_idx = torch.floor((x - cfg.vmin) / cfg.bin_size)
	bin_offset = ((x - cfg.vmin) / cfg.bin_size - bin_idx).unsqueeze(-1)
	soft_two_hot = torch.zeros(x.shape[0], cfg.num_bins, device=x.device, dtype=x.dtype)
	bin_idx = bin_idx.long()
	soft_two_hot = soft_two_hot.scatter(1, bin_idx.unsqueeze(1), 1 - bin_offset)
	soft_two_hot = soft_two_hot.scatter(1, (bin_idx.unsqueeze(1) + 1) % cfg.num_bins, bin_offset)

	# print(f"two_hot_debug: x={x.shape},nan:{x.isnan().any()} bin_idx={bin_idx.shape},nan:{bin_idx.isnan().any()} bin_offset={bin_offset.shape},nan:{bin_offset.isnan().any()} soft_two_hot={soft_two_hot.shape},nan:{soft_two_hot.isnan().any()}")

	return soft_two_hot


def _soft_ce_debug(pred, target, cfg):
	#print(f"_soft_ce_debug: orig_pred={pred.shape}, nan:{pred.isnan().any()} orig_target={target.shape}, nan:{target.isnan().any()}")
	pred = F.log_softmax(pred, dim=-1)
	target = two_hot_debug(target, cfg)
	#print(f"_soft_ce_debug: pred={pred.shape}, nan:{pred.isnan().any()} target={target.shape}, nan:{target.isnan().any()}")
	return -(target * pred).sum(-1, keepdim=True)

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

		self._prev_logits = torch.nn.Buffer(torch.zeros(self.cfg.horizon, self.cfg.action_dim, device=self.device))

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
		action_onehot, info = self.model.pi(z, task)

		if eval_mode:
			action = info["mean"]

			assert action.shape == (1,), f"Unexpected shape: {action.shape}"

		# action_onehot = action[0].cpu()

		assert action_onehot.shape == (self.cfg.action_dim,), f"Unexpected action_onehot shape: {action_onehot.shape}"

		return action_onehot

	@torch.no_grad()
	def _estimate_value(self, z, actions, task):
		"""Estimate value of a trajectory starting at latent state z and executing given actions."""

		#print(f"_estimate_value: actions.shape={actions.shape}")

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
		action_onehot, _ = self.model.pi(z, task)

		#print(f"INPUTS TO Q USED, in _estimate_value: z.shape={z.shape} action_onehot={action_onehot.shape}")
		result = self.model.Q(z, action_onehot, task, return_type='avg')
		#print(f"RESULT IN _estimate_value after model.Q: {result.shape}")

		return G + discount * (1-termination) * self.model.Q(z, action_onehot, task, return_type='avg')

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
		# Encode observation to latent
		z = self.model.encode(obs, task)

		# (Optional) Sample policy-guided trajectories
		if self.cfg.num_pi_trajs > 0:

			#print(f"OBS shape: {obs.shape}")

			pi_actions_onehot = torch.empty(
				self.cfg.horizon, self.cfg.num_pi_trajs, self.cfg.action_dim,
				dtype=torch.float, device=self.device
			)

			#print(f"pi_actions_onehot shape: {pi_actions_onehot.shape}")

			_z = z.repeat(self.cfg.num_pi_trajs, 1)

			#print(f"_z shape: {_z.shape}")

			for t in range(self.cfg.horizon - 1):
				action_onehot, _ = self.model.pi(_z, task)  # logits: (num_pi_trajs, action_dim)

				#print(f"action_onehot shape: {action_onehot.shape}")
				pi_actions_onehot[t] = action_onehot
				_z = self.model.next(_z, action_onehot, task)

			action_onehot, _ = self.model.pi(_z, task)
			pi_actions_onehot[-1] = action_onehot

		# Initialize state and parameters.  Expand latent for sampling.
		#print(f"Z INPUT TO QS, before repeat ({self.cfg.num_samples}): z.shape={z.shape}")
		z = z.repeat(self.cfg.num_samples, 1)
		#print(f"Z INPUT TO QS, after repeat ({self.cfg.num_samples}): z.shape={z.shape}")

		# Allocate discrete one-hot action tensor
		actions_onehot = torch.empty(self.cfg.horizon, self.cfg.num_samples, self.cfg.action_dim, device=self.device)
		# actions = torch.empty(self.cfg.horizon, self.cfg.num_samples, dtype=torch.long, device=self.device)

		# Insert policy-guided one-hot actions
		if self.cfg.num_pi_trajs > 0:
			actions_onehot[:, :self.cfg.num_pi_trajs] = pi_actions_onehot


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

		elif _USE_ONEHOT := True:
			# Initialize categorical logits for each timestep
			logits = torch.zeros(self.cfg.horizon, self.cfg.action_dim, device=self.device)

			if not t0:
				logits[:-1] = self._prev_logits[1:]

			# MPPI Loop
			for _ in range(self.cfg.iterations):
				# Sample random action indexes
				random_actions = torch.randint(
					low=0, high=self.cfg.action_dim,
					size=(self.cfg.horizon, self.cfg.num_samples - self.cfg.num_pi_trajs),
					device=self.device
				)
				random_onehot = F.one_hot(random_actions, num_classes=self.cfg.action_dim).float()

				# Fill remaining slots in actions tensor
				actions_onehot[:, self.cfg.num_pi_trajs:] = random_onehot

				if self.cfg.multitask:
					actions_onehot = actions_onehot * self.model._action_masks[task]

				#print(f"VALUE SHAPE input to _estimate_value: z.shape={z.shape} actions_onehot.shape={actions_onehot.shape}")

				# Estimate value for all samples
				value = self._estimate_value(z, actions_onehot, task).nan_to_num(0)  # (num_samples, 1)

				#print(f"VALUE SHAPE result of _estimate_value: {value.shape}")

				elite_idxs = torch.topk(value.squeeze(1), self.cfg.num_elites, dim=0).indices  # (num_elites,)
				elite_value = value[elite_idxs]  # (num_elites,)
				elite_actions = actions_onehot[:, elite_idxs]  # (horizon, num_elites, action_dim)

				# Score for softmax-weighted histogram
				score = torch.exp(self.cfg.temperature * (elite_value - elite_value.max()))
				score = score / (score.sum() + 1e-9)  # shape (E,)

				# Update logits as soft histogram (accumulate weighted one-hots)
				for t in range(self.cfg.horizon):
					# Weighted average over elite one-hots
					logits[t] = (score.view(-1, 1) * elite_actions[t]).sum(dim=0)  # (action_dim,)

			# Select final action from logits at t=0
			probs = torch.softmax(logits[0], dim=-1)
			if eval_mode:
				action_index = torch.argmax(probs)
			else:
				action_dist = torch.distributions.Categorical(probs)
				action_index = action_dist.sample()

			# Save for next step
			self._prev_logits.copy_(logits)

			# Return either one-hot or index depending on downstream usage
			#action_onehot = F.one_hot(action_index, num_classes=self.cfg.action_dim).float()

			#return action_onehot
			return action_index

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
		action_onehot, info = self.model.pi(zs, task)
		entropy = info["entropy"].unsqueeze(-1)

		#print(f"update_pi INPUTS TO model.Q: zs={zs.shape} action_onehot={action_onehot.shape}")

		qs = self.model.Q(zs, action_onehot, task, return_type='avg', detach=True)
		self.scale.update(qs[0])
		qs = self.scale(qs)

		#print(f"entropy shape: {entropy.shape}  qs.shape={qs.shape}")

		# Loss is a weighted sum of Q-values
		rho = torch.pow(self.cfg.rho, torch.arange(len(qs), device=self.device))
		#r0 = self.cfg.entropy_coef * entropy + qs
		#r1 = -r0.mean(dim=(1,2))
		#r2 = r1 * rho
		#r2.mean()

		pi_loss = (-(self.cfg.entropy_coef * entropy + qs).mean(dim=(1,2)) * rho).mean()
		pi_loss.backward()
		pi_grad_norm = torch.nn.utils.clip_grad_norm_(self.model._pi.parameters(), self.cfg.grad_clip_norm)
		self.pi_optim.step()
		self.pi_optim.zero_grad(set_to_none=True)

		info = TensorDict({
			"pi_loss": pi_loss,
			"pi_grad_norm": pi_grad_norm,
			"pi_entropy": info["entropy"],
			#"pi_scaled_entropy": info["scaled_entropy"],
			#"pi_scale": self.scale.value,
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
		action_onehot, _ = self.model.pi(next_z, task)
		discount = self.discount[task].unsqueeze(-1) if self.cfg.multitask else self.discount
		return reward + discount * (1-terminated) * self.model.Q(next_z, action_onehot, task, return_type='min', target=True)

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
		#b_action = action.unsqueeze(-1)

		# Predictions
		_zs = zs[:-1]

		#print(f"update_pi INPUTS TO model.Q: _zs={zs.shape} action_onehot={action.shape}")
		qs = self.model.Q(_zs, action, task, return_type='all')

		# print(f"qs: {qs.shape},nan:{qs.isnan().any()} _zs: {_zs.shape},nan:{_zs.isnan().any()}")

		reward_preds = self.model.reward(_zs, action, task)
		if self.cfg.episodic:
			termination_pred = self.model.termination(zs[1:], task, unnormalized=True)

		# Compute losses
		reward_loss, value_loss = 0, 0
		for t, (rew_pred_unbind, rew_unbind, td_targets_unbind, qs_unbind) in enumerate(zip(reward_preds.unbind(0), reward.unbind(0), td_targets.unbind(0), qs.unbind(1))):
			# print("REWARD LOSS DEBUG")
			# soft_ce_result = _soft_ce_debug(rew_pred_unbind, rew_unbind, self.cfg).mean()
			# print(f"REWARD LOSS UPDATE parts: rew_pred_unbind={rew_pred_unbind.shape} rew_unbind={rew_unbind.shape} rho={self.cfg.rho} soft_ce_result={soft_ce_result=} t={t=}")


			reward_loss = reward_loss + math.soft_ce(rew_pred_unbind, rew_unbind, self.cfg).mean() * self.cfg.rho**t
			for _, qs_unbind_unbind in enumerate(qs_unbind.unbind(0)):
				# print("VALUE LOSS DEBUG")
				# soft_ce_result = _soft_ce_debug(qs_unbind_unbind, td_targets_unbind, self.cfg).mean()
				# print(f"VALUE LOSS UPDATE parts: qs_unbind_unbind={qs_unbind_unbind.shape} td_targets_unbind={td_targets_unbind.shape} rho={self.cfg.rho} soft_ce_result={soft_ce_result=} t={t=}")

				value_loss = value_loss + math.soft_ce(qs_unbind_unbind, td_targets_unbind, self.cfg).mean() * self.cfg.rho**t

		if reward_loss.isnan().any() or value_loss.isnan().any():
			print("FOUND NAN")
			reward_nan = torch.isnan(reward_loss).nonzero(as_tuple=True)[0]
			value_nan = torch.isnan(value_loss).nonzero(as_tuple=True)[0]
			w_terminated = terminated.nonzero(as_tuple=True)[0]
			print(f"w_reward_isnan={reward_nan}  reward_loss.shape={reward_loss.shape}")
			print(f"w_value_isnan={value_nan} value_loss.shape={value_loss.shape}")
			print(f"w_terminated={w_terminated=} terminated.shape={terminated.shape}")
			print(f"terminated={terminated=}")
			print(f"reward_preds={reward_preds=}")
			print(f"reward_preds: nan:{reward_preds.isnan().any()} {reward_preds=}")
			print(f"reward: nan:{reward.isnan().any()} {reward=}")
			print(f"td_targets_unbind: nan:{td_targets_unbind.isnan().any()} {td_targets_unbind=}")
			print(f"qs: nan:{qs.isnan().any()} {qs=}")
			print(f"reward_loss: nan:{reward_loss.isnan().any()} {reward_loss=}")
			print(f"value_loss: nan:{value_loss.isnan().any()} {value_loss=}")
			raise AssertionError("FOUND NAN")

		# print(f"_update(): reward_loss0={reward_loss} value_loss={value_loss}")

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
		obs, action_index, reward, terminated, task = buffer.sample()

		action_onehot = F.one_hot(action_index, num_classes=self.cfg.action_dim)

		# print(f"ACTION SHAPE FROM BUFFER: {action.shape}")

		# action = action.unsqueeze(-1)

		kwargs = {}
		if task is not None:
			kwargs["task"] = task
		torch.compiler.cudagraph_mark_step_begin()

		# print(f"SHAPE PASSED IN TO _update: {action.shape}")

		return self._update(obs, action_onehot, reward, terminated, **kwargs)

	def get_actor_policy_probs_and_critic_value(self, obs):
		# (1, 512)
		z = self.model.encode(obs, task=None)

		# (1, 7)
		logits = self.model._pi(z)

		# (1, 7)
		action_probs = logits

		# print(f"action_probs.shape: {action_probs.shape}")

		# Convert every action to one-hot encoding
		# (7, 7)
		all_actions = F.one_hot(torch.tensor(list(range(self.cfg.action_dim))), num_classes=7).float().to(device=self.device)

		# print(f"Z DIM: {z.shape} obs={obs.shape}")

		# Repeat observation for each action.
		z_repeated = z.expand(7, -1)      # (7, 512) — no actual memory duplication

		# print(f"Z REPEATED: {z_repeated.shape} all_actions={all_actions.shape}")

		if False:
			# -> (5, 7, 101)
			qs = []
			for action_index in range(self.cfg.action_dim):
				if True:
					# WORKS
					action_onehot = F.one_hot(torch.tensor(action_index), num_classes=self.cfg.action_dim).float().to(self.device)
					print(f"Z.shape: {z.shape} action_onehot.shape={action_onehot.shape}")

					self._estimate_value(z_repeated, all_actions)
					qs_all = self.model.Q(z, action_onehot.unsqueeze(0), task=None, return_type='all', detach=True)

					# Average all of the Q values from the ensemble.
					qs = qs_all.mean(dim=0)

					q_result = qs
				else:
					# WORKS TOO
					print(f"z_repeated.shape: {z_repeated.shape} all_actions.shape={all_actions.shape}")
					q_result = self.model.Q(z_repeated, all_actions, task=None, return_type='all', detach=True)

				print(f"Q RESULT SHAPE: {q_result.shape}")
				qs.append(q_result)

		# print(f"GET model.Q z_repeated.shape={z_repeated.shape} all_actions.shape={all_actions.shape}")
		all_values = self.model.Q(z_repeated, all_actions, task=None, return_type='avg')
		# print(f"GET model.Q all_values.shape={all_values.shape}")
		# all_values = self._estimate_value(z_repeated, all_actions.unsqueeze(0), task=None)
		value = all_values.mean(dim=0).unsqueeze(0)

		return action_probs, value

	def get_action_probs(self, obs):
		z = self.model.encode(obs, task=None)
		action_onehot, info = self.model.pi(z, task=None)
		logits = info['logits']
		probs = Categorical(logits=logits)
		return probs.probs
