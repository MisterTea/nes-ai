#!/usr/bin/env python3
# Adapted heavily from:
#   https://docs.cleanrl.dev/rl-algorithms/ppo-rnd/#ppo_rnd_envpoolpy
#   Docs and experiment results can be found at https://docs.cleanrl.dev/rl-algorithms/ppo-rnd/#ppo_rnd_envpoolpy

import os
import random
import time
from collections import deque
from dataclasses import dataclass
from datetime import datetime

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import tyro
from gym.wrappers.normalize import RunningMeanStd
from torch.distributions.categorical import Categorical
from torch.utils.tensorboard import SummaryWriter
from four_rooms_env import OBS_W, OBS_H, FourRoomsEnv

from gymnasium.envs.registration import register

register(
    id="FourRooms-v0",
    entry_point=FourRoomsEnv,
    max_episode_steps=1_000_000,
)


NdArrayUint8 = np.ndarray[np.dtype[np.uint8]]


@dataclass
class Args:
    r"""
    Run example:
        > WANDB_API_KEY=<key> python3 ppo_nes.py --wandb-project-name mariorl --track

        ...
        wandb: Tracking run with wandb version 0.19.9
        wandb: Run data is saved locally in /Users/dave/rl/nes-ai/wandb/run-20250418_130130-SuperMarioBros-v0__ppo_nes__1__2025-04-18_13-01-30
        wandb: Run `wandb offline` to turn off syncing.
        wandb: Syncing run SuperMarioBros-v0__ppo_nes__1__2025-04-18_13-01-30
        wandb: ⭐️ View project at https://wandb.ai/millman-none/mariorl
        wandb: 🚀 View run at https://wandb.ai/millman-none/mariorl/runs/SuperMarioBros-v0__ppo_nes__1__2025-04-18_13-01-30

    Resume example:
        > WANDB_API_KEY=<key> WANDB_RUN_ID=SuperMarioBros-v0__ppo_nes__1__2025-04-18_13-01-30 WANDB_RESUME=must python3 ppo_nes.py --wandb-project-name mariorl --track

        ...
        wandb: Tracking run with wandb version 0.19.9
        wandb: Run data is saved locally in /Users/dave/rl/nes-ai/wandb/run-20250418_133317-SuperMarioBros-v0__ppo_nes__1__2025-04-18_13-01-30
        wandb: Run `wandb offline` to turn off syncing.
    --> wandb: Resuming run SuperMarioBros-v0__ppo_nes__1__2025-04-18_13-01-30
        wandb: ⭐️ View project at https://wandb.ai/millman-none/mariorl
        wandb: 🚀 View run at https://wandb.ai/millman-none/mariorl/runs/SuperMarioBros-v0__ppo_nes__1__2025-04-18_13-01-30
        ...
    --> resumed at update 9
        ...
    """

    exp_name: str = os.path.basename(__file__)[: -len(".py")]
    """the name of this experiment"""
    seed: int = 1
    """seed of the experiment"""
    torch_deterministic: bool = True
    """if toggled, `torch.backends.cudnn.deterministic=False`"""
    cuda: bool = True
    """if toggled, cuda will be enabled by default"""
    track: bool = False
    """if toggled, this experiment will be tracked with Weights and Biases"""
    wandb_project_name: str = "FourRoomsRL"
    """the wandb's project name"""
    wandb_entity: str | None = None
    """the entity (team) of wandb's project"""
    wandb_run_id: str | None = None
    """the id of a wandb run to resume"""
    capture_video: bool = False
    """whether to capture videos of the agent performances (check out `videos` folder)"""
    checkpoint_frequency: float = 30
    """create a checkpoint every N seconds"""
    train_agent: bool = True
    """enable or disable training of the agent"""

    # Visualization
    value_sweep_frequency: int | None = 0

    visualize_decoder: bool = True

    """create a value sweep visualization every N updates"""
    visualize_reward: bool = True
    visualize_actions: bool = True
    visualize_intrinsic_decoder: bool = True
    visualize_intrinsic_reward: bool = True

    # Specific experiments
    dump_trajectories: bool = False
    reset_to_save_state: bool = False

    # Algorithm specific arguments
    env_id: str = "FourRooms-v0"
    """the id of the environment"""
    total_timesteps: int = 10000000
    """total timesteps of the experiments"""
    learning_rate: float = 2.5e-4
    """the learning rate of the optimizer"""
    learning_rate_decoder: float = 2.5e-4
    """the learning rate of the decoder optimizer"""
    num_envs: int = 1
    """the number of parallel game environments"""
    num_steps: int = 256
    """the number of steps to run in each environment per policy rollout"""
    anneal_lr: bool = False
    """Toggle learning rate annealing for policy and value networks"""
    gamma: float = 0.99
    """the discount factor gamma"""
    gae_lambda: float = 0.95
    """the lambda for the general advantage estimation"""
    num_minibatches: int = 4
    """the number of mini-batches"""
    update_epochs: int = 4
    """the K epochs to update the policy"""
    norm_adv: bool = True
    """Toggles advantages normalization"""
    clip_coef: float = 0.1
    """the surrogate clipping coefficient"""
    clip_vloss: bool = True
    """Toggles whether or not to use a clipped loss for the value function, as per the paper."""
    ent_coef: float = 0.1
    """coefficient of the entropy"""
    vf_coef: float = 0.5
    """coefficient of the value function"""
    max_grad_norm: float = 0.5
    """the maximum norm for the gradient clipping"""
    target_kl: float = None
    """the target KL divergence threshold"""
    intrinsic_reward_coef: float = 1e-3
    """coefficient of the intrinsic reward when combining with extrinsic reward"""

    # RND arguments
    update_proportion: float = 0.25
    """proportion of exp used for predictor update"""
    int_coef: float = 1.0
    """coefficient of intrinsic reward"""
    ext_coef: float = 2.0
    """coefficient of extrinsic reward"""
    int_gamma: float = 0.99
    """Intrinsic reward discount rate"""
    num_iterations_obs_norm_init: int = 10
    """number of iterations to initialize the observations normalization parameters"""

    # to be filled in runtime
    batch_size: int = 0
    """the batch size (computed in runtime)"""
    minibatch_size: int = 0
    """the mini-batch size (computed in runtime)"""
    num_iterations: int = 0
    """the number of iterations (computed in runtime)"""


def make_env(env_id, idx, capture_video, run_name):
    def thunk():
        if capture_video and idx == 0:
            env = gym.make(env_id, render_mode="rgb_array")
            env = gym.wrappers.RecordVideo(env, f"videos/{run_name}")
        else:
            env = gym.make(env_id, render_mode="human")
        env = gym.wrappers.RecordEpisodeStatistics(env)
        env = gym.wrappers.FrameStackObservation(env, 1)
        return env

    return thunk


def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer


class Agent(nn.Module):
    def __init__(self, envs):
        super().__init__()
        self.network = nn.Sequential(
            nn.Flatten(),                     # (batch, 7, 7) -> (batch, 49)
            layer_init(nn.Linear(49, 256)),
            nn.ReLU(),
            layer_init(nn.Linear(256, 448)),
            nn.ReLU(),
        )

        self.extra_layer = nn.Sequential(layer_init(nn.Linear(448, 448), std=0.1), nn.ReLU())
        self.actor = nn.Sequential(
            layer_init(nn.Linear(448, 448), std=0.01),
            nn.ReLU(),
            layer_init(nn.Linear(448, envs.single_action_space.n), std=0.01),
        )
        self.critic_ext = layer_init(nn.Linear(448, 1), std=0.01)
        self.critic_int = layer_init(nn.Linear(448, 1), std=0.01)

    def get_action_and_value(self, x, action=None):
        if False:
            print(f"get_action_and_value: dtype={x.dtype} min={x.min()} max={x.max()}")
            assert not torch.isnan(x).any(), "Observation has NaNs!"

        hidden = self.network(x / 255.0)

        if False:
            print("Hidden min:", hidden.min().item(), "max:", hidden.max().item(), "any nan:", torch.isnan(hidden).any().item())

        logits = self.actor(hidden)

        if False:
            print("Actor logits min:", logits.min().item(), "max:", logits.max().item(), "any nan:", torch.isnan(logits).any().item())

            for name, param in self.actor.named_parameters():
                if torch.isnan(param).any():
                    print(f"NaN detected in actor parameter: {name}")

        probs = Categorical(logits=logits)
        features = self.extra_layer(hidden)
        if action is None:
            action = probs.sample()
        return (
            action,
            probs.log_prob(action),
            probs.entropy(),
            self.critic_ext(features + hidden),
            self.critic_int(features + hidden),
        )

    def get_value(self, x):
        hidden = self.network(x / 255.0)
        features = self.extra_layer(hidden)
        return self.critic_ext(features + hidden), self.critic_int(features + hidden)


class RNDModel(nn.Module):
    def __init__(self, input_size, output_size):
        super().__init__()

        self.input_size = input_size
        self.output_size = output_size

        feature_output = 256

        # Prediction network
        self.predictor = nn.Sequential(
            nn.Flatten(),                     # (batch, 7, 7) -> (batch, 49)
            layer_init(nn.Linear(49, 256)),
            nn.ReLU(),

            layer_init(nn.Linear(feature_output, 512)),
            nn.ReLU(),
            layer_init(nn.Linear(512, 512)),
            nn.ReLU(),
            layer_init(nn.Linear(512, 512)),
        )

        # Target network
        self.target = nn.Sequential(
            nn.Flatten(),                     # (batch, 7, 7) -> (batch, 49)
            layer_init(nn.Linear(49, 256)),
            nn.ReLU(),

            layer_init(nn.Linear(feature_output, 512)),
        )

        # target network is not trainable
        for param in self.target.parameters():
            param.requires_grad = False

    def forward(self, next_obs):
        target_feature = self.target(next_obs)
        predict_feature = self.predictor(next_obs)

        return predict_feature, target_feature


class RewardForwardFilter:
    def __init__(self, gamma):
        self.rewems = None
        self.gamma = gamma

    def update(self, rews):
        if self.rewems is None:
            self.rewems = rews
        else:
            self.rewems = self.rewems * self.gamma + rews
        return self.rewems


def main():
    args = tyro.cli(Args)

    # Derived args.
    args.batch_size = int(args.num_envs * args.num_steps)
    args.minibatch_size = int(args.batch_size // args.num_minibatches)
    args.num_iterations = args.total_timesteps // args.batch_size

    date_str = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

    # NOTE: Run name should be descriptive, but not unique.
    # In particular, we don't include the date because the date does not affect the results.
    # Date prefixes are handled by wandb automatically.

    if not args.wandb_run_id:
        run_prefix = f"{args.env_id}__{args.exp_name}__{args.seed}"
        run_name = f"{run_prefix}__{date_str}"
        args.wandb_run_id = run_name

    run_name = args.wandb_run_id

    if args.track:
        import wandb

        run = wandb.init(
            project=args.wandb_project_name,
            entity=args.wandb_entity,
            sync_tensorboard=True,
            config=vars(args),
            #name=run_name,
            monitor_gym=True,
            save_code=True,
            id=run_name,
        )
        assert run.dir == f"runs/{run_name}"
        run_dir = run.dir
    else:
        run_dir = f"runs/{run_name}"

    writer = SummaryWriter(f"runs/{run_name}")
    writer.add_text(
        "hyperparameters",
        "|param|value|\n|-|-|\n%s" % ("\n".join([f"|{key}|{value}|" for key, value in vars(args).items()])),
    )

    # TRY NOT TO MODIFY: seeding
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = args.torch_deterministic

    device = torch.device("cuda" if torch.cuda.is_available() and args.cuda else "cpu")

    if device == torch.device("cpu"):
        # Try mps
        if torch.backends.mps.is_available():
            device = torch.device("mps")
        else:
            print("No GPU available, using CPU.")

    # env setup
    envs = gym.vector.SyncVectorEnv(
        [make_env(args.env_id, i, args.capture_video, run_name) for i in range(args.num_envs)],
    )
    assert isinstance(envs.single_action_space, gym.spaces.Discrete), "only discrete action space is supported"

    # Virtual frame stacking.
    # envs_single_action_space_shape =(4,) + envs.single_action_space.shape
    envs_single_action_space_shape = envs.single_action_space.shape

    # ActorCritic
    agent = Agent(envs).to(device)
    rnd_model = RNDModel(4, envs.single_action_space.n).to(device)
    combined_parameters = list(agent.parameters()) + list(rnd_model.predictor.parameters())
    optimizer = optim.Adam(
        combined_parameters,
        lr=args.learning_rate,
        eps=1e-5,
    )

    reward_rms = RunningMeanStd()
    obs_rms = RunningMeanStd(shape=(1, 1, OBS_H, OBS_W))
    discounted_reward = RewardForwardFilter(args.int_gamma)

    if True:
        print(f"envs.single_observation_space.shape: {envs.single_observation_space.shape}")
        # Virtual frame stacking.

        # envs_single_observation_space_shape = (4,) + envs.single_observation_space.shape
        envs_single_observation_space_shape = envs.single_observation_space.shape

        print(f"envs.single_observation_space.shape after stack: {envs_single_observation_space_shape}")


    # ALGO Logic: Storage setup
    obs = torch.zeros((args.num_steps, args.num_envs) + envs_single_observation_space_shape).to(device)
    actions = torch.zeros((args.num_steps, args.num_envs) + envs_single_action_space_shape).to(device)
    logprobs = torch.zeros((args.num_steps, args.num_envs)).to(device)
    rewards = torch.zeros((args.num_steps, args.num_envs)).to(device)
    curiosity_rewards = torch.zeros((args.num_steps, args.num_envs)).to(device)
    dones = torch.zeros((args.num_steps, args.num_envs)).to(device)
    ext_values = torch.zeros((args.num_steps, args.num_envs)).to(device)
    int_values = torch.zeros((args.num_steps, args.num_envs)).to(device)
    avg_returns = deque(maxlen=20)

    # TRY NOT TO MODIFY: start the game
    global_step = 0
    start_time = time.time()
    next_obs, _info = envs.reset(seed=args.seed)

    # Virtual frame stack: repeat 4 times
    print(f"INIT NEXT OBS: {next_obs.shape}")

    next_obs = torch.Tensor(next_obs).to(device)
    next_done = torch.zeros(args.num_envs).to(device)
    num_updates = args.total_timesteps // args.batch_size

    print(f"OBS STARTING SHAPE: {obs.shape}")

    print("Start to initialize observation normalization parameter.....")
    next_ob = []
    for step in range(args.num_steps * args.num_iterations_obs_norm_init):
        acs = np.random.randint(0, envs.single_action_space.n, size=(args.num_envs,))
        s, _reward, d, _truncated, _info = envs.step(acs)

        if False:
            print(f"NEXT OBS: {next_obs.shape}")
            # print(f"NEXT OBS RESHAPED: {next_obs[:, -1, :, :].reshape(args.num_envs, 1, OBS_H, OBS_W) }")

        reshaped_ob = s[:, -1, :, :].reshape([-1, 1, OBS_H, OBS_W]).tolist()

        next_ob += reshaped_ob

        if len(next_ob) % (args.num_steps * args.num_envs) == 0:
            next_ob = np.stack(next_ob)
            obs_rms.update(next_ob)
            next_ob = []
    print("End to initialize...")

    next_obs = torch.Tensor(next_obs).to(device)
    next_done = torch.zeros(args.num_envs).to(device)

    if args.track and run.resumed:
        # Reference example (that seems out of date) from: https://docs.cleanrl.dev/advanced/resume-training/#resume-training_1
        # Updated with example from: https://wandb.ai/lavanyashukla/save_and_restore/reports/Saving-and-Restoring-Machine-Learning-Models-with-W-B--Vmlldzo3MDQ3Mw

        starting_iter = run.starting_step
        global_step = starting_iter * args.batch_size
        model = run.restore('files/agent.ckpt')

        agent.load_state_dict(torch.load(model.name, map_location=device))

        agent.eval()

        print(f"Resumed at update {starting_iter}")
    else:
        starting_iter = 1

    # Initialize last checkpoint time.  We don't want to checkpoint again for a bit.
    last_checkpoint_time = time.time()

    for update in range(1, num_updates + 1):
        # Annealing the rate if instructed to do so.
        if args.anneal_lr:
            frac = 1.0 - (update - 1.0) / num_updates
            lrnow = frac * args.learning_rate
            optimizer.param_groups[0]["lr"] = lrnow

        steps_start = time.time()

        for step in range(0, args.num_steps):
            global_step += 1 * args.num_envs
            obs[step] = next_obs
            dones[step] = next_done

            # ALGO LOGIC: action logic
            with torch.no_grad():
                if False:
                    print(f"UPDATE: {update} STEP: {step}")
                    print(f"OBS SHAPE: {obs.shape} min={obs.min()} max={obs.max()}")
                    print(f"OBS[step] SHAPE: {obs[step].shape} min={obs[step].min()} max={obs[step].max()}")

                value_ext, value_int = agent.get_value(obs[step])
                ext_values[step], int_values[step] = (
                    value_ext.flatten(),
                    value_int.flatten(),
                )
                action, logprob, _, _, _ = agent.get_action_and_value(obs[step])

            actions[step] = action
            logprobs[step] = logprob

            # TRY NOT TO MODIFY: execute the game and log data.
            next_obs, reward, done, _truncated, info = envs.step(action.cpu().numpy())
            rewards[step] = torch.tensor(reward, dtype=torch.float32).to(device).view(-1)
            next_obs, next_done = torch.Tensor(next_obs).to(device), torch.Tensor(done).to(device)

            if False:
                print(f"NEXT OBS: {next_obs.shape}")
                print(f"NEXT OBS RESHAPED: {next_obs[:, -1, :, :].reshape(args.num_envs, 1, OBS_H, OBS_W) }")

            rehsaped_obs = next_obs[:, -1, :, :].reshape(args.num_envs, 1, OBS_H, OBS_W)

            rnd_next_obs = (
                (
                    (rehsaped_obs - torch.from_numpy(obs_rms.mean.astype(np.float32)).to(device))
                    / torch.sqrt(torch.from_numpy(obs_rms.var.astype(np.float32)).to(device))
                ).clip(-5, 5)
            ).float()

            if False:
                print(f"rnd_next_obs: {rnd_next_obs.shape} has_nan: {torch.isnan(rnd_next_obs).any()}")

            target_next_feature = rnd_model.target(rnd_next_obs)
            predict_next_feature = rnd_model.predictor(rnd_next_obs)
            curiosity_rewards[step] = ((target_next_feature - predict_next_feature).pow(2).sum(1) / 2).data
            for idx, d in enumerate(done):
                if d:
                    episodic_return = info["episodic_return"][idx]

                    avg_returns.append(episodic_return)
                    epi_ret = np.average(avg_returns)
                    print(
                        f"global_step={global_step}, episodic_return={episodic_return}, curiosity_reward={np.mean(curiosity_rewards[step].cpu().numpy())}"
                    )
                    writer.add_scalar("charts/avg_episodic_return", epi_ret, global_step)
                    writer.add_scalar("charts/episodic_return", episodic_return, global_step)
                    writer.add_scalar(
                        "charts/episode_curiosity_reward",
                        curiosity_rewards[step][idx],
                        global_step,
                    )
                    writer.add_scalar("charts/episodic_length", info["episodic_length"][idx], global_step)

        curiosity_reward_per_env = np.array(
            [discounted_reward.update(reward_per_step) for reward_per_step in curiosity_rewards.cpu().data.numpy().T]
        )
        mean, std, count = (
            np.mean(curiosity_reward_per_env),
            np.std(curiosity_reward_per_env),
            len(curiosity_reward_per_env),
        )
        reward_rms.update_from_moments(mean, std**2, count)

        curiosity_rewards /= np.sqrt(reward_rms.var)

        steps_end = time.time()

        if args.train_agent:
            optimize_networks_start = time.time()

            # bootstrap value if not done
            with torch.no_grad():
                next_value_ext, next_value_int = agent.get_value(next_obs)
                next_value_ext, next_value_int = next_value_ext.reshape(1, -1), next_value_int.reshape(1, -1)
                ext_advantages = torch.zeros_like(rewards, device=device)
                int_advantages = torch.zeros_like(curiosity_rewards, device=device)
                ext_lastgaelam = 0
                int_lastgaelam = 0
                for t in reversed(range(args.num_steps)):
                    if t == args.num_steps - 1:
                        ext_nextnonterminal = 1.0 - next_done
                        int_nextnonterminal = 1.0
                        ext_nextvalues = next_value_ext
                        int_nextvalues = next_value_int
                    else:
                        ext_nextnonterminal = 1.0 - dones[t + 1]
                        int_nextnonterminal = 1.0
                        ext_nextvalues = ext_values[t + 1]
                        int_nextvalues = int_values[t + 1]
                    ext_delta = rewards[t] + args.gamma * ext_nextvalues * ext_nextnonterminal - ext_values[t]
                    int_delta = curiosity_rewards[t] + args.int_gamma * int_nextvalues * int_nextnonterminal - int_values[t]
                    ext_advantages[t] = ext_lastgaelam = (
                        ext_delta + args.gamma * args.gae_lambda * ext_nextnonterminal * ext_lastgaelam
                    )
                    int_advantages[t] = int_lastgaelam = (
                        int_delta + args.int_gamma * args.gae_lambda * int_nextnonterminal * int_lastgaelam
                    )
                ext_returns = ext_advantages + ext_values
                int_returns = int_advantages + int_values

            # flatten the batch
            b_obs = obs.reshape((-1,) + envs_single_observation_space_shape)
            b_logprobs = logprobs.reshape(-1)
            b_actions = actions.reshape(-1)
            b_ext_advantages = ext_advantages.reshape(-1)
            b_int_advantages = int_advantages.reshape(-1)
            b_ext_returns = ext_returns.reshape(-1)
            b_int_returns = int_returns.reshape(-1)
            b_ext_values = ext_values.reshape(-1)

            b_advantages = b_int_advantages * args.int_coef + b_ext_advantages * args.ext_coef

            reshaped_b_obs = b_obs[:, -1, :, :].reshape(-1, 1, OBS_H, OBS_W)

            obs_rms.update(reshaped_b_obs.cpu().numpy())

            executed_epochs = 0
            epochs_start = time.time()

            # Optimizing the policy and value network
            b_inds = np.arange(args.batch_size)

            reshaped_rnd_next_obs = b_obs[:, -1, :, :].reshape(-1, 1, OBS_H, OBS_W)

            rnd_next_obs = (
                (
                    (reshaped_rnd_next_obs - torch.from_numpy(obs_rms.mean.astype(np.float32)).to(device))
                    / torch.sqrt(torch.from_numpy(obs_rms.var.astype(np.float32)).to(device))
                ).clip(-5, 5)
            ).float()

            clipfracs = []
            for epoch in range(args.update_epochs):
                executed_epochs += 1

                np.random.shuffle(b_inds)
                for start in range(0, args.batch_size, args.minibatch_size):
                    end = start + args.minibatch_size
                    mb_inds = b_inds[start:end]

                    predict_next_state_feature, target_next_state_feature = rnd_model(rnd_next_obs[mb_inds])
                    forward_loss = F.mse_loss(
                        predict_next_state_feature, target_next_state_feature.detach(), reduction="none"
                    ).mean(-1)

                    mask = torch.rand(len(forward_loss), device=device)
                    mask = (mask < args.update_proportion).type(torch.FloatTensor).to(device)
                    forward_loss = (forward_loss * mask).sum() / torch.max(
                        mask.sum(), torch.tensor([1], device=device, dtype=torch.float32)
                    )

                    if False:
                        print(f"get_action_and_value, b_obs[0].shape: {b_obs[0].shape}")
                        assert not torch.isnan(b_obs[mb_inds].any()), "Found nans!"

                    _, newlogprob, entropy, new_ext_values, new_int_values = agent.get_action_and_value(
                        b_obs[mb_inds], b_actions.long()[mb_inds]
                    )
                    logratio = newlogprob - b_logprobs[mb_inds]
                    ratio = logratio.exp()

                    with torch.no_grad():
                        # calculate approx_kl http://joschu.net/blog/kl-approx.html
                        old_approx_kl = (-logratio).mean()
                        approx_kl = ((ratio - 1) - logratio).mean()
                        clipfracs += [((ratio - 1.0).abs() > args.clip_coef).float().mean().item()]

                    mb_advantages = b_advantages[mb_inds]

                    if args.norm_adv:
                        #print(f"MB ADVANTAGES: mean={mb_advantages.mean()} std={mb_advantages.std()}")
                        #mb_advantages = (mb_advantages - mb_advantages.mean()) / (mb_advantages.std() + 1e-8)
                        #print(f"MB ADVANTAGES nan: {torch.isnan(mb_advantages).any()}")

                        std = mb_advantages.std()
                        if std < 1e-8 or torch.isnan(std):
                            mb_advantages_norm = mb_advantages - mb_advantages.mean()  # just mean-center
                        else:
                            mb_advantages_norm = (mb_advantages - mb_advantages.mean()) / (std + 1e-8)

                        mb_advantages = mb_advantages_norm

                    # Policy loss
                    pg_loss1 = -mb_advantages * ratio
                    pg_loss2 = -mb_advantages * torch.clamp(ratio, 1 - args.clip_coef, 1 + args.clip_coef)
                    pg_loss = torch.max(pg_loss1, pg_loss2).mean()

                    if False:
                        print(f"PG LOSS VALUES: pg_loss={pg_loss} pg_loss1={pg_loss1} pg_loss2={pg_loss2} mb_advantages: {mb_advantages}")

                    # Value loss
                    new_ext_values, new_int_values = new_ext_values.view(-1), new_int_values.view(-1)
                    if args.clip_vloss:
                        ext_v_loss_unclipped = (new_ext_values - b_ext_returns[mb_inds]) ** 2
                        ext_v_clipped = b_ext_values[mb_inds] + torch.clamp(
                            new_ext_values - b_ext_values[mb_inds],
                            -args.clip_coef,
                            args.clip_coef,
                        )
                        ext_v_loss_clipped = (ext_v_clipped - b_ext_returns[mb_inds]) ** 2
                        ext_v_loss_max = torch.max(ext_v_loss_unclipped, ext_v_loss_clipped)
                        ext_v_loss = 0.5 * ext_v_loss_max.mean()
                    else:
                        ext_v_loss = 0.5 * ((new_ext_values - b_ext_returns[mb_inds]) ** 2).mean()

                    int_v_loss = 0.5 * ((new_int_values - b_int_returns[mb_inds]) ** 2).mean()
                    v_loss = ext_v_loss + int_v_loss
                    entropy_loss = entropy.mean()
                    loss = pg_loss - args.ent_coef * entropy_loss + v_loss * args.vf_coef + forward_loss

                    if False:
                        print(f"LOSS VALUES: loss={loss} pg_loss={pg_loss} entropy_loss={entropy_loss} v_loss={v_loss} forward_loss={forward_loss}")

                    optimizer.zero_grad()
                    loss.backward()
                    if args.max_grad_norm:
                        nn.utils.clip_grad_norm_(
                            combined_parameters,
                            args.max_grad_norm,
                        )
                    optimizer.step()

                if args.target_kl is not None:
                    if approx_kl > args.target_kl:
                        break

            epochs_end = time.time()
            epoch_dt = epochs_end - epochs_start

            optimize_networks_end = time.time()

            num_samples = executed_epochs * args.batch_size
            per_sample_dt = epoch_dt / num_samples

            steps_dt = steps_end - steps_start
            optimize_networks_dt = optimize_networks_end - optimize_networks_start

            print(f"Time steps: (num_steps={args.num_steps}): {steps_dt:.4f}")
            print(f"Time optimize: (epochs={args.update_epochs} batch_size={args.batch_size} minibatch_size={args.minibatch_size}) per-sample: {per_sample_dt:.4f} optimize_networks: {optimize_networks_dt:.4f}")

            # TRY NOT TO MODIFY: record rewards for plotting purposes
            writer.add_scalar("charts/learning_rate", optimizer.param_groups[0]["lr"], global_step)
            writer.add_scalar("losses/value_loss", v_loss.item(), global_step)
            writer.add_scalar("losses/policy_loss", pg_loss.item(), global_step)
            writer.add_scalar("losses/entropy", entropy_loss.item(), global_step)
            writer.add_scalar("losses/old_approx_kl", old_approx_kl.item(), global_step)
            writer.add_scalar("losses/fwd_loss", forward_loss.item(), global_step)
            writer.add_scalar("losses/approx_kl", approx_kl.item(), global_step)
            print("SPS:", int(global_step / (time.time() - start_time)))
            writer.add_scalar("charts/SPS", int(global_step / (time.time() - start_time)), global_step)

        # Checkpoint.
        seconds_since_last_checkpoint = time.time() - last_checkpoint_time
        if args.checkpoint_frequency > 0 and seconds_since_last_checkpoint > args.checkpoint_frequency:
            seconds_since_last_checkpoint = time.time() - last_checkpoint_time
            print(f"Checkpoint at iter: {update}, since last checkpoint: {seconds_since_last_checkpoint:.2f}s")
            start_checkpoint = time.time()

            # NOTE: The run.dir location includes a 'files/' suffix.
            #
            # E.g. 'agent.cpkt' will be saved to:
            #   /Users/dave/rl/nes-ai/wandb/run-20250418_130130-SuperMarioBros-v0__ppo_nes__1__2025-04-18_13-01-30/files/agent.ckpt
            #
            torch.save(agent.state_dict(), f"{run_dir}/agent.ckpt")

            if args.track:
                wandb.save(f"{run_dir}/agent.ckpt", policy="now")

            print(f"Checkpoint done: {time.time() - start_checkpoint:.4f}s")

            # Reset the checkpoint time, so we don't include the amount of time necessary to perform
            # the checkpoint itself.
            last_checkpoint_time = time.time()

    envs.close()
    writer.close()


if __name__ == "__main__":
    main()