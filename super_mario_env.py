# docs and experiment results can be found at https://docs.cleanrl.dev/rl-algorithms/ppo/#ppo_ataripy
from typing import Optional

import gymnasium as gym
import numpy as np
import torch

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
        if _PRINT_FRAME_INFO := False:
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
class SuperMarioEnv(gym.Env):

    def __init__(self):
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
     entry_point=SuperMarioEnv,
     max_episode_steps=60 * 60 * 5,
)
