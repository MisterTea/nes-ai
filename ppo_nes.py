# docs and experiment results can be found at https://docs.cleanrl.dev/rl-algorithms/ppo/#ppo_ataripy
import os
import random
import time
from dataclasses import dataclass
from typing import Optional

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import tyro
from torch.distributions.categorical import Categorical
from torch.utils.tensorboard import SummaryWriter

from stable_baselines3.common.atari_wrappers import (  # isort:skip
    ClipRewardEnv,
    EpisodicLifeEnv,
    FireResetEnv,
    MaxAndSkipEnv,
    NoopResetEnv,
)

from nes import NES, SYNC_AUDIO, SYNC_NONE, SYNC_PYGAME, SYNC_VSYNC
from nes_ai.ai.base import SELECT, START, compute_reward_map

class NesAle:
    def __init__(self):
        pass

    def lives(self):
        return 1


class SimpleAiHandler:
    def __init__(self):
        print("INIT AI HANDLER")
        self.frame_num = -1
        self.controller1_pressed_state = np.zeros(shape=(8,), dtype=np.uint8)
        self.screen_buffer_image = np.zeros((3, 224, 224))

        self.last_reward_map = None
        self.reward_map = None

        self.prev_info = None

        # From: nes/peripherals.py:323:
        #   self.is_pressed[self.A] = int(state[self.A])
        #   self.is_pressed[self.B] = int(state[self.B])
        #   self.is_pressed[self.SELECT] = int(state[self.SELECT])
        #   self.is_pressed[self.START] = int(state[self.START])
        #   self.is_pressed[self.UP] = int(state[self.UP])
        #   self.is_pressed[self.DOWN] = int(state[self.DOWN])
        #   self.is_pressed[self.LEFT] = int(state[self.LEFT])
        #   self.is_pressed[self.RIGHT] = int(state[self.RIGHT])
        self.controller_state_desc = ["A", "B", "SELECT", "START", "UP", "DOWN", "LEFT", "RIGHT"]

    def _describe_controller_vector(self, is_pressed) -> str:
        pressed = [
            desc
            for is_button_pressed, desc in zip(is_pressed, self.controller_state_desc)
            if is_button_pressed
        ]
        return str(pressed)


    def shutdown(self):
        print("Shutting down ai handler")

    def update(self, frame: int, controller1, ram, screen_buffer_image):
        self.reward_map, reward_vector = compute_reward_map(
            self.last_reward_map, torch.from_numpy(ram).int()
        )
        # if self.frames_until_exit > 0:
        #     self.frames_until_exit -= 1
        #     if self.frames_until_exit == 0:
        #         print("REACHED END")
        #         return False
        # else:
        #     if frame > 200:
        #         if self.reward_map.lives < 2:
        #             print("LOST LIFE")
        #             self.frames_until_exit = 10
        #         if self.reward_map.level > 0:
        #             print("GOT NEW LEVEL")
        #             self.frames_until_exit = 300

        always_changing_info = f"Frame: {frame:<5} Time left: {self.reward_map.time_left:<5} "
        controller_desc = self._describe_controller_vector(controller1.is_pressed)
        new_info = f"Left pos: {self.reward_map.left_pos:<5} Reward: {self.reward_map} {reward_vector} Controller: {controller_desc}"
        if new_info == self.prev_info:
            # Clear out old line and display again.
            #print("", end='\r', flush=True)
            print(always_changing_info + new_info, end='\r', flush=True)
        else:
            # New info, start a new line.
            print('\n' + always_changing_info + new_info, end='\r', flush=True)


        self.prev_info = new_info

        if False:
            print(f"FRAME: {frame}")
            print(
                "TIME LEFT", self.reward_map.time_left, "LEFT POS", self.reward_map.left_pos
            )
            # minimum_progress = (400 - self.reward_map.time_left) * 2
            # if frame > 200 and self.reward_map.left_pos < minimum_progress:
            #     print("TOO LITTLE PROGRESS, DYING")
            #     ram[0x75A] = 1
            # print("FALLING IN PIT", torch.from_numpy(ram).int()[0x0712])
            print("REWARD", self.reward_map, reward_vector)

            print("CONTROLLER", controller1.is_pressed)

        self.frame_num = frame
        self.controller1_pressed_state = controller1.is_pressed
        self.screen_buffer_image = screen_buffer_image

        # print(f"SCREEN BUFFER SHAPE: {self.screen_buffer_image.shape}, sum={self.screen_buffer_image.sum()}")

        # with torch.no_grad():
        #     # from torchvision import transforms

        #     image = DEFAULT_TRANSFORM(screen_buffer_image)
        #     # print("Environment Image ", frame, image.mean())
        #     for buffer_i in range(3):
        #         self.screen_buffer[buffer_i, :, :, :] = self.screen_buffer[
        #             buffer_i + 1, :, :, :
        #         ]
        #     self.screen_buffer[3, :, :, :] = image[:, :, :]

        #     if self.learn_mode == LearnMode.REPLAY_IMITATION:
        #         # Replay
        #         controller = self.rollout_data.expert_controller[str(frame)]
        #         controller1.set_state(controller)
        #         assert (
        #             self.rollout_data.reward_map_history[str(frame)] == self.reward_map
        #         ), f"{self.rollout_data.reward_map_history[str(frame)]} != {self.reward_map}"
        #         print("REWARD VECTOR", reward_vector)
        #         self.rollout_data.reward_vector_history[str(frame)] = reward_vector

        #     if self.learn_mode == LearnMode.IMITATION_VALIDATION:
        #         # Replay
        #         controller = self.bootstrap_expert_data.expert_controller[str(frame)]
        #         assert isinstance(controller, list), f"{type(controller)}"
        #         controller1.set_state(controller)
        #         assert (
        #             self.bootstrap_expert_data.reward_map_history[str(frame)]
        #             == self.reward_map
        #         ), f"{self.bootstrap_expert_data.reward_map_history[str(frame)]} != {self.reward_map}"
        #         print("REWARD VECTOR", reward_vector)
        #         self.rollout_data.reward_vector_history[str(frame)] = reward_vector

        #         if frame >= 200:
        #             # Score model timm

        #             print(self.screen_buffer.shape, image.shape)
        #             reward_history = self._get_reward_history(frame)

        #             action, log_prob, entropy, value = score(
        #                 self.score_model,
        #                 self.screen_buffer,
        #                 self.controller_buffer,
        #                 reward_history,
        #                 str(frame),
        #             )

        #             if frame >= 205:  # Check against ground truth
        #                 (
        #                     image_stack,
        #                     value,
        #                     past_inputs,
        #                     past_rewards,
        #                     label_int,
        #                     recorded_action_log_prob,
        #                     advantages,
        #                 ) = self.expert_dataset.get_frame(frame)
        #                 assert torch.equal(
        #                     past_inputs, self.controller_buffer
        #                 ), f"{past_inputs} != {self.controller_buffer}"
        #                 if not torch.equal(image_stack, self.screen_buffer):
        #                     print(image_stack[3].mean(), self.screen_buffer[3].mean())
        #                     assert torch.equal(
        #                         image_stack[0], self.screen_buffer[0]
        #                     ), f"{image_stack[0]} != {self.screen_buffer[0]}"
        #                     assert torch.equal(
        #                         image_stack[1], self.screen_buffer[1]
        #                     ), f"{image_stack[1]} != {self.screen_buffer[1]}"
        #                     assert torch.equal(
        #                         image_stack[2], self.screen_buffer[2]
        #                     ), f"{image_stack[2]} != {self.screen_buffer[2]}"
        #                     assert torch.equal(
        #                         image_stack[3], self.screen_buffer[3]
        #                     ), f"{image_stack[3]} != {self.screen_buffer[3]}"

        #                 logged_action, logged_log_prob, logged_entropy, logged_value = (
        #                     score(
        #                         self.score_model,
        #                         image_stack,
        #                         past_inputs,
        #                         past_rewards,
        #                         str(frame),
        #                     )
        #                 )
        #                 if not torch.equal(action, logged_action):
        #                     print(f"Actions don't match: {action} != {logged_action}")
        #                 else:
        #                     assert torch.equal(
        #                         log_prob, logged_log_prob
        #                     ), f"{log_prob} != {logged_log_prob}"
        #                     assert torch.equal(
        #                         entropy, logged_entropy
        #                     ), f"{entropy} != {logged_entropy}"

        #             self.rollout_data.agent_params[str(frame)] = {
        #                 "action": action.tolist(),
        #                 "log_prob": float(log_prob.item()),
        #                 "entropy": float(entropy.item()),
        #                 "value": value.tolist(),
        #             }
        #             controller1.update()
        #             if not controller1.is_any_pressed():
        #                 print(action)
        #                 controller1.set_state(action)

        #             # Overwrite with expert
        #             expert_controller_for_frame = (
        #                 self.bootstrap_expert_data.expert_controller_no_start_select(
        #                     str(frame)
        #                 )
        #             )
        #             if not torch.equal(
        #                 torch.IntTensor(expert_controller_for_frame).to(
        #                     device=action.device
        #                 ),
        #                 action,
        #             ):
        #                 print("WRONG ANSWER")
        #                 controller1.set_state(expert_controller_for_frame)

        #         else:
        #             # Use expert for intro
        #             controller = self.bootstrap_expert_data.expert_controller[
        #                 str(frame)
        #             ]
        #             controller1.set_state(controller)

        #     if self.learn_mode == LearnMode.RL:
        #         if frame == 0:
        #             self.rollout_data.clear()

        #         if frame >= 200:
        #             # Score model timm
        #             reward_history = self._get_reward_history(frame)

        #             action, log_prob, entropy, value = score(
        #                 self.score_model,
        #                 self.screen_buffer,
        #                 self.controller_buffer,
        #                 reward_history,
        #                 str(frame),
        #             )
        #             print("LOGPROB", log_prob)
        #             self.rollout_data.agent_params[str(frame)] = {
        #                 "action": action.cpu().tolist(),
        #                 "log_prob": float(log_prob.item()),
        #                 "entropy": float(entropy.item()),
        #                 "value": value.cpu().tolist(),
        #                 # "screen_buffer": self.screen_buffer.cpu().tolist(),
        #                 "controller_buffer": self.controller_buffer.cpu().tolist(),
        #                 "reward_history": reward_history.cpu().tolist(),
        #             }
        #             controller1.update()
        #             if not controller1.is_any_pressed():
        #                 # Human isn't taking over
        #                 controller1.set_state(action.tolist())

        #         else:
        #             # Use expert for intro
        #             controller = self.bootstrap_expert_data.expert_controller[
        #                 str(frame)
        #             ]
        #             controller1.set_state(controller)

        #     if (
        #         self.learn_mode == LearnMode.DATA_COLLECT
        #         or self.learn_mode == LearnMode.IMITATION_VALIDATION
        #         or self.learn_mode == LearnMode.RL
        #     ):
        #         if frame == 0:
        #             print("RESETTING")
        #             self.rollout_data.input_images.clear()
        #             self.rollout_data.expert_controller.clear()
        #             self.rollout_data.reward_map_history.clear()
        #             self.rollout_data.reward_vector_history.clear()
        #             self.rollout_data.agent_params.clear()

        #         # Data Collect
        #         controller_pressed = controller1.get_ai_state()
        #         self.rollout_data.expert_controller[str(frame)] = controller_pressed
        #         self.rollout_data.reward_map_history[str(frame)] = self.reward_map
        #         self.rollout_data.reward_vector_history[str(frame)] = reward_vector
        #         self.rollout_data.put_image(screen_buffer_image, frame)
        #         if frame % 60 == 0:
        #             print("SAVING")
        #             self.rollout_data.sync()

        #     for buffer_i in range(2):
        #         self.controller_buffer[buffer_i, :] = self.controller_buffer[
        #             buffer_i + 1, :
        #         ]
        #     self.controller_buffer[2, :] = torch.FloatTensor(controller1.get_ai_state())

        return True

    # def _get_reward_history(self, frame):
    #     reward_history = torch.zeros((3, REWARD_VECTOR_SIZE), dtype=torch.float)
    #     for x in range(3):
    #         reward_history[x, :] = torch.FloatTensor(
    #             self.rollout_data.reward_vector_history[str((frame - 3) + x)]
    #         )
    #     return reward_history

# https://gymnasium.farama.org/introduction/create_custom_env/
class NesEnv(gym.Env):

    def __init__(self, rom_path: str = "SuperMarioBros"):
        # From: nes/peripherals.py:323:
        #   self.is_pressed[self.A] = int(state[self.A])
        #   self.is_pressed[self.B] = int(state[self.B])
        #   self.is_pressed[self.SELECT] = int(state[self.SELECT])
        #   self.is_pressed[self.START] = int(state[self.START])
        #   self.is_pressed[self.UP] = int(state[self.UP])
        #   self.is_pressed[self.DOWN] = int(state[self.DOWN])
        #   self.is_pressed[self.LEFT] = int(state[self.LEFT])
        #   self.is_pressed[self.RIGHT] = int(state[self.RIGHT])

        self.action_space = gym.spaces.Discrete(8)


        # From: https://gymnasium.farama.org/api/spaces/fundamental/
        #
        #   Nintendo Game Controller - Can be conceptualized as 3 discrete action spaces:
        #     Arrow Keys: Discrete 5 - NOOP[0], UP[1], RIGHT[2], DOWN[3], LEFT[4] - params: min: 0, max: 4
        #     Button A: Discrete 2 - NOOP[0], Pressed[1] - params: min: 0, max: 1
        #     Button B: Discrete 2 - NOOP[0], Pressed[1] - params: min: 0, max: 1

        # self.action_space = gym.spaces.MultiDiscrete([ 5, 2, 2 ])

        # From: nes/ai_handler.py:34
        # self.screen_buffer = torch.zeros((4, 3, 224, 224), dtype=torch.float)

        self.observation_space = gym.spaces.Box(low=0, high=255, shape=(224, 224, 3), dtype=np.uint8)

        # self.size = size  # The size of the square grid
        # self.window_size = 512  # The size of the PyGame window

        # # Observations are dictionaries with the agent's and the target's location.
        # # Each location is encoded as an element of {0, ..., `size`}^2, i.e. MultiDiscrete([size, size]).
        # self.observation_space = spaces.Dict(
        #     {
        #         "agent": spaces.Box(0, size - 1, shape=(2,), dtype=int),
        #         "target": spaces.Box(0, size - 1, shape=(2,), dtype=int),
        #     }
        # )

        self.ale = NesAle()

        # Initialize the NES Emulator.
        self.nes = NES(
            "./roms/Super_mario_brothers.nes",
            SimpleAiHandler(),
            sync_mode=SYNC_PYGAME,
            opengl=True,
            audio=False,
        )
        self.ai_handler = self.nes.ai_handler

    def _get_obs(self):
        # Get screen buffer.  Shape = (3, 244, 244)
        screen_buffer = np.array(self.ai_handler.screen_buffer_image)

        if False:
            print(f"SCREEN BUFFER DTYPE: {type(screen_buffer)} dtype={screen_buffer.dtype}")
            print(f"SCREEN BUFFER: shape={screen_buffer.shape}")
            print(f"SCREEN BUFFER: min={screen_buffer.min()}, max={screen_buffer.max()}")

        # Convert to expected input size.  Shape = (244, 244, 3)
        # obs = (screen_buffer.T * 255).astype(np.uint8)
        obs = screen_buffer

        # return np.ones((224, 224, 3), dtype=np.uint8)
        return obs

    def _get_info(self):
        # return {
        #     "distance": np.linalg.norm(
        #         self._agent_location - self._target_location, ord=1
        #     )
        # }
        return {}

    def reset(self, seed: Optional[int] = None, options: Optional[dict] = None):
        # We need the following line to seed self.np_random
        # super().reset(seed=seed)

        # # Choose the agent's location uniformly at random
        # self._agent_location = self.np_random.integers(0, self.size, size=2, dtype=int)

        # # We will sample the target's location randomly until it does not coincide with the agent's location
        # self._target_location = self._agent_location
        # while np.array_equal(self._target_location, self._agent_location):
        #     self._target_location = self.np_random.integers(
        #         0, self.size, size=2, dtype=int
        #     )

        # print(f"WHATS ON NES: {dir(self.nes)}")

        # Reset CPU and controller.
        self.nes.reset()
        self.nes.run_init()

        # Take a single step.
        self.nes.run_frame()

        # Get initial values.
        observation = self._get_obs()
        info = self._get_info()

        print(f"OBSERVATION FROM RESET: {observation.shape}")

        return observation, info

    def step(self, action):
        # print("STEPPING")

        # Apply actions to controller.
        action = np.zeros(8, dtype=np.uint8)

        # Start button.
        action[3] = 1

        # Right button.
        action[7] = 1
        action = 7

        # Take a step in the emulator.
        self.nes.run_frame()

        # Read off the current reward.


        # Map the action (element of {0,1,2,3}) to the direction we walk in
        # direction = self._action_to_direction[action]
        # # We use `np.clip` to make sure we don't leave the grid bounds
        # self._agent_location = np.clip(
        #     self._agent_location + direction, 0, self.size - 1
        # )

        # # An environment is completed if and only if the agent has reached the target
        # terminated = np.array_equal(self._agent_location, self._target_location)
        # truncated = False
        # reward = 1 if terminated else 0  # the agent is only reached at the end of the episode
        reward = 0
        terminated = False
        truncated = False
        observation = self._get_obs()
        info = self._get_info()

        return observation, reward, terminated, truncated, info

    def render(self):
        if self.render_mode == "rgb_array":
            return self._render_frame()
        else:
            return None

    def _render_frame(self):
        # if self.window is None and self.render_mode == "human":
        #     pygame.init()
        #     pygame.display.init()
        #     self.window = pygame.display.set_mode(
        #         (self.window_size, self.window_size)
        #     )
        # if self.clock is None and self.render_mode == "human":
        #     self.clock = pygame.time.Clock()

        # canvas = pygame.Surface((self.window_size, self.window_size))
        # canvas.fill((255, 255, 255))
        # pix_square_size = (
        #     self.window_size / self.size
        # )  # The size of a single grid square in pixels

        # # First we draw the target
        # pygame.draw.rect(
        #     canvas,
        #     (255, 0, 0),
        #     pygame.Rect(
        #         pix_square_size * self._target_location,
        #         (pix_square_size, pix_square_size),
        #     ),
        # )
        # # Now we draw the agent
        # pygame.draw.circle(
        #     canvas,
        #     (0, 0, 255),
        #     (self._agent_location + 0.5) * pix_square_size,
        #     pix_square_size / 3,
        # )

        # # Finally, add some gridlines
        # for x in range(self.size + 1):
        #     pygame.draw.line(
        #         canvas,
        #         0,
        #         (0, pix_square_size * x),
        #         (self.window_size, pix_square_size * x),
        #         width=3,
        #     )
        #     pygame.draw.line(
        #         canvas,
        #         0,
        #         (pix_square_size * x, 0),
        #         (pix_square_size * x, self.window_size),
        #         width=3,
        #     )

        # if self.render_mode == "human":
        #     # The following line copies our drawings from `canvas` to the visible window
        #     self.window.blit(canvas, canvas.get_rect())
        #     pygame.event.pump()
        #     pygame.display.update()

        #     # We need to ensure that human-rendering occurs at the predefined framerate.
        #     # The following line will automatically add a delay to keep the framerate stable.
        #     self.clock.tick(self.metadata["render_fps"])
        # else:  # rgb_array
        #     return np.transpose(
        #         np.array(pygame.surfarray.pixels3d(canvas)), axes=(1, 0, 2)
        #     )
        pass


from gymnasium.envs.registration import register

register(
     id="SuperMarioBros-v0",
     entry_point=NesEnv,
     max_episode_steps=60 * 60 * 5,
)


@dataclass
class Args:
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
    wandb_project_name: str = "cleanRL"
    """the wandb's project name"""
    wandb_entity: str = None
    """the entity (team) of wandb's project"""
    capture_video: bool = False
    """whether to capture videos of the agent performances (check out `videos` folder)"""

    # Algorithm specific arguments
    env_id: str = "SuperMarioBros-v0"
    """the id of the environment"""
    total_timesteps: int = 10000000
    """total timesteps of the experiments"""
    learning_rate: float = 2.5e-4
    """the learning rate of the optimizer"""
    num_envs: int = 1
    """the number of parallel game environments"""
    num_steps: int = 128
    """the number of steps to run in each environment per policy rollout"""
    anneal_lr: bool = True
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
    ent_coef: float = 0.01
    """coefficient of the entropy"""
    vf_coef: float = 0.5
    """coefficient of the value function"""
    max_grad_norm: float = 0.5
    """the maximum norm for the gradient clipping"""
    target_kl: float = None
    """the target KL divergence threshold"""

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
            env = gym.make(env_id)
        env = gym.wrappers.RecordEpisodeStatistics(env)
        #env = NoopResetEnv(env, noop_max=30)
        env = MaxAndSkipEnv(env, skip=4)
        env = EpisodicLifeEnv(env)
        # if "FIRE" in env.unwrapped.get_action_meanings():
        #     env = FireResetEnv(env)
        env = ClipRewardEnv(env)
        env = gym.wrappers.ResizeObservation(env, (84, 84))
        env = gym.wrappers.GrayscaleObservation(env)
        env = gym.wrappers.FrameStackObservation(env, 4)
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
            layer_init(nn.Conv2d(4, 32, 8, stride=4)),
            nn.ReLU(),
            layer_init(nn.Conv2d(32, 64, 4, stride=2)),
            nn.ReLU(),
            layer_init(nn.Conv2d(64, 64, 3, stride=1)),
            nn.ReLU(),
            nn.Flatten(),
            layer_init(nn.Linear(64 * 7 * 7, 512)),
            nn.ReLU(),
        )
        self.actor = layer_init(nn.Linear(512, envs.single_action_space.n), std=0.01)
        self.critic = layer_init(nn.Linear(512, 1), std=1)

    def get_value(self, x):
        return self.critic(self.network(x / 255.0))

    def get_action_and_value(self, x, action=None):
        hidden = self.network(x / 255.0)
        logits = self.actor(hidden)
        probs = Categorical(logits=logits)
        if action is None:
            action = probs.sample()
        return action, probs.log_prob(action), probs.entropy(), self.critic(hidden)


if __name__ == "__main__":
    args = tyro.cli(Args)
    args.batch_size = int(args.num_envs * args.num_steps)
    args.minibatch_size = int(args.batch_size // args.num_minibatches)
    args.num_iterations = args.total_timesteps // args.batch_size
    run_name = f"{args.env_id}__{args.exp_name}__{args.seed}__{int(time.time())}"
    if args.track:
        import wandb

        wandb.init(
            project=args.wandb_project_name,
            entity=args.wandb_entity,
            sync_tensorboard=True,
            config=vars(args),
            name=run_name,
            monitor_gym=True,
            save_code=True,
        )
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

    # env setup
    envs = gym.vector.SyncVectorEnv(
        [make_env(args.env_id, i, args.capture_video, run_name) for i in range(args.num_envs)],
    )
    assert isinstance(envs.single_action_space, gym.spaces.Discrete), "only discrete action space is supported"

    agent = Agent(envs).to(device)
    optimizer = optim.Adam(agent.parameters(), lr=args.learning_rate, eps=1e-5)

    # ALGO Logic: Storage setup
    obs = torch.zeros((args.num_steps, args.num_envs) + envs.single_observation_space.shape).to(device)
    actions = torch.zeros((args.num_steps, args.num_envs) + envs.single_action_space.shape).to(device)
    logprobs = torch.zeros((args.num_steps, args.num_envs)).to(device)
    rewards = torch.zeros((args.num_steps, args.num_envs)).to(device)
    dones = torch.zeros((args.num_steps, args.num_envs)).to(device)
    values = torch.zeros((args.num_steps, args.num_envs)).to(device)

    # TRY NOT TO MODIFY: start the game
    global_step = 0
    start_time = time.time()
    next_obs, _ = envs.reset(seed=args.seed)

    print(f"NEXT OBS SHAPE: {next_obs.shape}")

    next_obs = torch.Tensor(next_obs).to(device)
    next_done = torch.zeros(args.num_envs).to(device)

    for iteration in range(1, args.num_iterations + 1):
        print(f"ITER: {iteration}")
        # Annealing the rate if instructed to do so.
        if args.anneal_lr:
            frac = 1.0 - (iteration - 1.0) / args.num_iterations
            lrnow = frac * args.learning_rate
            optimizer.param_groups[0]["lr"] = lrnow

        for step in range(0, args.num_steps):
            # print(f"STEP: {step}")
            global_step += args.num_envs
            obs[step] = next_obs
            dones[step] = next_done

            # ALGO LOGIC: action logic
            with torch.no_grad():
                action, logprob, _, value = agent.get_action_and_value(next_obs)
                values[step] = value.flatten()
            actions[step] = action
            logprobs[step] = logprob

            # TRY NOT TO MODIFY: execute the game and log data.
            next_obs, reward, terminations, truncations, infos = envs.step(action.cpu().numpy())
            next_done = np.logical_or(terminations, truncations)
            rewards[step] = torch.tensor(reward).to(device).view(-1)
            next_obs, next_done = torch.Tensor(next_obs).to(device), torch.Tensor(next_done).to(device)

            if "final_info" in infos:
                for info in infos["final_info"]:
                    if info and "episode" in info:
                        print(f"global_step={global_step}, episodic_return={info['episode']['r']}")
                        writer.add_scalar("charts/episodic_return", info["episode"]["r"], global_step)
                        writer.add_scalar("charts/episodic_length", info["episode"]["l"], global_step)

        # bootstrap value if not done
        with torch.no_grad():
            next_value = agent.get_value(next_obs).reshape(1, -1)
            advantages = torch.zeros_like(rewards).to(device)
            lastgaelam = 0
            for t in reversed(range(args.num_steps)):
                if t == args.num_steps - 1:
                    nextnonterminal = 1.0 - next_done
                    nextvalues = next_value
                else:
                    nextnonterminal = 1.0 - dones[t + 1]
                    nextvalues = values[t + 1]
                delta = rewards[t] + args.gamma * nextvalues * nextnonterminal - values[t]
                advantages[t] = lastgaelam = delta + args.gamma * args.gae_lambda * nextnonterminal * lastgaelam
            returns = advantages + values

        print("DONE ITERATIONS")

        # flatten the batch
        b_obs = obs.reshape((-1,) + envs.single_observation_space.shape)
        b_logprobs = logprobs.reshape(-1)
        b_actions = actions.reshape((-1,) + envs.single_action_space.shape)
        b_advantages = advantages.reshape(-1)
        b_returns = returns.reshape(-1)
        b_values = values.reshape(-1)

        # Optimizing the policy and value network
        b_inds = np.arange(args.batch_size)
        clipfracs = []
        for epoch in range(args.update_epochs):
            np.random.shuffle(b_inds)
            for start in range(0, args.batch_size, args.minibatch_size):
                end = start + args.minibatch_size
                mb_inds = b_inds[start:end]

                _, newlogprob, entropy, newvalue = agent.get_action_and_value(b_obs[mb_inds], b_actions.long()[mb_inds])
                logratio = newlogprob - b_logprobs[mb_inds]
                ratio = logratio.exp()

                with torch.no_grad():
                    # calculate approx_kl http://joschu.net/blog/kl-approx.html
                    old_approx_kl = (-logratio).mean()
                    approx_kl = ((ratio - 1) - logratio).mean()
                    clipfracs += [((ratio - 1.0).abs() > args.clip_coef).float().mean().item()]

                mb_advantages = b_advantages[mb_inds]
                if args.norm_adv:
                    mb_advantages = (mb_advantages - mb_advantages.mean()) / (mb_advantages.std() + 1e-8)

                # Policy loss
                pg_loss1 = -mb_advantages * ratio
                pg_loss2 = -mb_advantages * torch.clamp(ratio, 1 - args.clip_coef, 1 + args.clip_coef)
                pg_loss = torch.max(pg_loss1, pg_loss2).mean()

                # Value loss
                newvalue = newvalue.view(-1)
                if args.clip_vloss:
                    v_loss_unclipped = (newvalue - b_returns[mb_inds]) ** 2
                    v_clipped = b_values[mb_inds] + torch.clamp(
                        newvalue - b_values[mb_inds],
                        -args.clip_coef,
                        args.clip_coef,
                    )
                    v_loss_clipped = (v_clipped - b_returns[mb_inds]) ** 2
                    v_loss_max = torch.max(v_loss_unclipped, v_loss_clipped)
                    v_loss = 0.5 * v_loss_max.mean()
                else:
                    v_loss = 0.5 * ((newvalue - b_returns[mb_inds]) ** 2).mean()

                entropy_loss = entropy.mean()
                loss = pg_loss - args.ent_coef * entropy_loss + v_loss * args.vf_coef

                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(agent.parameters(), args.max_grad_norm)
                optimizer.step()

            if args.target_kl is not None and approx_kl > args.target_kl:
                break

        y_pred, y_true = b_values.cpu().numpy(), b_returns.cpu().numpy()
        var_y = np.var(y_true)
        explained_var = np.nan if var_y == 0 else 1 - np.var(y_true - y_pred) / var_y

        # TRY NOT TO MODIFY: record rewards for plotting purposes
        writer.add_scalar("charts/learning_rate", optimizer.param_groups[0]["lr"], global_step)
        writer.add_scalar("losses/value_loss", v_loss.item(), global_step)
        writer.add_scalar("losses/policy_loss", pg_loss.item(), global_step)
        writer.add_scalar("losses/entropy", entropy_loss.item(), global_step)
        writer.add_scalar("losses/old_approx_kl", old_approx_kl.item(), global_step)
        writer.add_scalar("losses/approx_kl", approx_kl.item(), global_step)
        writer.add_scalar("losses/clipfrac", np.mean(clipfracs), global_step)
        writer.add_scalar("losses/explained_variance", explained_var, global_step)
        print("SPS:", int(global_step / (time.time() - start_time)))
        writer.add_scalar("charts/SPS", int(global_step / (time.time() - start_time)), global_step)

    envs.close()
    writer.close()
