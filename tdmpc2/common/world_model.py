from copy import deepcopy

import torch
import torch.nn as nn
import torch.nn.functional as F

from . import layers, math, init
from tensordict import TensorDict
from tensordict.nn import TensorDictParams
from torch.distributions.categorical import Categorical


class WorldModel(nn.Module):
    """
    TD-MPC2 implicit world model architecture.
    Can be used for both single-task and multi-task experiments.
    """

    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        if cfg.multitask:
            self._task_emb = nn.Embedding(len(cfg.tasks), cfg.task_dim, max_norm=1)
            self.register_buffer("_action_masks", torch.zeros(len(cfg.tasks), cfg.action_dim))
            for i in range(len(cfg.tasks)):
                self._action_masks[i, :cfg.action_dims[i]] = 1.
        self._encoder = layers.enc(cfg)
        self._dynamics = layers.mlp(cfg.latent_dim + cfg.action_dim + cfg.task_dim, 2*[cfg.mlp_dim], cfg.latent_dim, act=layers.SimNorm(cfg))
        self._reward = layers.mlp(cfg.latent_dim + cfg.action_dim + cfg.task_dim, 2*[cfg.mlp_dim], max(cfg.num_bins, 1))
        self._termination = layers.mlp(cfg.latent_dim + cfg.task_dim, 2*[cfg.mlp_dim], 1) if cfg.episodic else None
        self._pi = layers.mlp(cfg.latent_dim + cfg.task_dim, 2*[cfg.mlp_dim], cfg.action_dim)
        self._Qs = layers.Ensemble([layers.mlp(cfg.latent_dim + cfg.action_dim + cfg.task_dim, 2*[cfg.mlp_dim], max(cfg.num_bins, 1), dropout=cfg.dropout) for _ in range(cfg.num_q)])
        self.apply(init.weight_init)
        init.zero_([self._reward[-1].weight, self._Qs.params["2", "weight"]])

        self.register_buffer("log_std_min", torch.tensor(cfg.log_std_min))
        self.register_buffer("log_std_dif", torch.tensor(cfg.log_std_max) - self.log_std_min)
        self.init()

    def init(self):
        # Create params
        self._detach_Qs_params = TensorDictParams(self._Qs.params.data, no_convert=True)
        self._target_Qs_params = TensorDictParams(self._Qs.params.data.clone(), no_convert=True)

        # Create modules
        with self._detach_Qs_params.data.to("meta").to_module(self._Qs.module):
            self._detach_Qs = deepcopy(self._Qs)
            self._target_Qs = deepcopy(self._Qs)

        # Assign params to modules
        # We do this strange assignment to avoid having duplicated tensors in the state-dict -- working on a better API for this
        delattr(self._detach_Qs, "params")
        self._detach_Qs.__dict__["params"] = self._detach_Qs_params
        delattr(self._target_Qs, "params")
        self._target_Qs.__dict__["params"] = self._target_Qs_params

    def __repr__(self):
        repr = 'TD-MPC2 World Model\n'
        modules = ['Encoder', 'Dynamics', 'Reward', 'Termination', 'Policy prior', 'Q-functions']
        for i, m in enumerate([self._encoder, self._dynamics, self._reward, self._termination, self._pi, self._Qs]):
            if m == self._termination and not self.cfg.episodic:
                continue
            repr += f"{modules[i]}: {m}\n"
        repr += "Learnable parameters: {:,}".format(self.total_params)
        return repr

    @property
    def total_params(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def to(self, *args, **kwargs):
        super().to(*args, **kwargs)
        self.init()
        return self

    def train(self, mode=True):
        """
        Overriding `train` method to keep target Q-networks in eval mode.
        """
        super().train(mode)
        self._target_Qs.train(False)
        return self

    def soft_update_target_Q(self):
        """
        Soft-update target Q-networks using Polyak averaging.
        """
        self._target_Qs_params.lerp_(self._detach_Qs_params, self.cfg.tau)

    def task_emb(self, x, task):
        """
        Continuous task embedding for multi-task experiments.
        Retrieves the task embedding for a given task ID `task`
        and concatenates it to the input `x`.
        """
        if isinstance(task, int):
            task = torch.tensor([task], device=x.device)
        emb = self._task_emb(task.long())
        if x.ndim == 3:
            emb = emb.unsqueeze(0).repeat(x.shape[0], 1, 1)
        elif emb.shape[0] == 1:
            emb = emb.repeat(x.shape[0], 1)
        return torch.cat([x, emb], dim=-1)

    def encode(self, obs, task):
        """
        Encodes an observation into its latent representation.
        This implementation assumes a single state-based observation.
        """

        if self.cfg.multitask:
            obs = self.task_emb(obs, task)

        # print(f"CONFIG OBS: {self.cfg.obs=} ndim={obs.ndim}")
        # print(f"encode obs: {obs.shape} ndim={obs.ndim}")
        if self.cfg.obs == 'rgb' and obs.ndim == 5:
            result_obs = torch.stack([self._encoder[self.cfg.obs](o) for o in obs])
            # print(f"RESULT_OBS: {result_obs.shape}")
            return result_obs
        return self._encoder[self.cfg.obs](obs)

    def next(self, z, a_onehot, task):
        """
        Predicts the next latent state given the current latent state and action.
        """
        if self.cfg.multitask:
            z = self.task_emb(z, task)

        assert a_onehot.shape[-1] == self.cfg.action_dim, f"Unexpected action shape: {a_onehot.shape}"
        # print(f"z.shape={z.shape} a.shape={a_onehot.shape}")

        z = torch.cat([z, a_onehot], dim=-1)
        return self._dynamics(z)

    def reward(self, z, a_onehot, task):
        """
        Predicts instantaneous (single-step) reward.
        """
        if self.cfg.multitask:
            z = self.task_emb(z, task)

        assert a_onehot.shape[-1] == self.cfg.action_dim, f"Unexpected action shape: {a_onehot.shape}"
        z = torch.cat([z, a_onehot], dim=-1)
        return self._reward(z)

    def termination(self, z, task, unnormalized=False):
        """
        Predicts termination signal.
        """
        assert task is None
        if self.cfg.multitask:
            z = self.task_emb(z, task)
        if unnormalized:
            return self._termination(z)
        return torch.sigmoid(self._termination(z))


    def pi(self, z, task):
        """
        Samples an action from the policy prior.
        The policy prior is a Gaussian distribution with
        mean and (log) std predicted by a neural network.
        """
        if self.cfg.multitask:
            z = self.task_emb(z, task)

        #print(f"z in _pi: {z.shape}")
        logits = self._pi(z)
        #print(f"logits in _pi: {logits.shape}")

        if self.cfg.multitask:
            logits = logits * self._action_masks[task]

        dist = Categorical(logits=logits)
        action = dist.sample()
        info = TensorDict({
            "logits": logits,
            "log_prob": dist.log_prob(action),
            "entropy": dist.entropy(),
        })
        #print(f"pi result action: {action.shape}")
        action_onehot = F.one_hot(action, num_classes=self.cfg.action_dim)

        #print(f"pi logits={logits.shape} dist={dist=} action={action.shape} action_onehot={action_onehot.shape}")

        return action_onehot, info

    def Q(self, z, a_onehot, task, return_type='min', target=False, detach=False):
        """
        Predict state-action value.
        `return_type` can be one of [`min`, `avg`, `all`]:
            - `min`: return the minimum of two randomly subsampled Q-values.
            - `avg`: return the average of two randomly subsampled Q-values.
            - `all`: return all Q-values.
        `target` specifies whether to use the target Q-networks or not.
        """
        assert return_type in {'min', 'avg', 'all'}

        if self.cfg.multitask:
            z = self.task_emb(z, task)

        # assert z.shape[:-1] == a.shape[:-1], f"Mismatched action shape: z.shape={z.shape} a.shape={a.shape}"
        assert a_onehot.shape[-1] == self.cfg.action_dim, f"Unexpected action shape: {a_onehot.shape}"
        z = torch.cat([z, a_onehot], dim=-1)

        if target:
            qnet = self._target_Qs
        elif detach:
            qnet = self._detach_Qs
        else:
            qnet = self._Qs
        out = qnet(z)

        if return_type == 'all':
            return out

        qidx = torch.randperm(self.cfg.num_q, device=out.device)[:2]
        Q = math.two_hot_inv(out[qidx], self.cfg)
        if return_type == "min":
            return Q.min(0).values
        return Q.sum(0) / 2
