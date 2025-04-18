# docs and experiment results can be found at https://docs.cleanrl.dev/rl-algorithms/ppo/#ppo_ataripy
from typing import Literal
from typing import Optional

import gymnasium as gym
import numpy as np
import torch

from nes import NES, SYNC_PYGAME
from nes_ai.ai.base import RewardMap, compute_reward_map

NdArrayUint8 = np.ndarray[tuple[Literal[4]], np.dtype[np.uint8]]

class NesAle:
    """
    For compatibility with EpisodicLifeEnv.

    Without, we'll see errors when adding EpisodicLifeEnv:

    Traceback (most recent call last):
        ...
        File "/Users/dave/rl/nes-ai/venv/lib/python3.11/site-packages/stable_baselines3/common/atari_wrappers.py", line 145, in reset
        self.lives = self.env.unwrapped.ale.lives()  # type: ignore[attr-defined]
                    ^^^^^^^^^^^^^^^^^^^^^^
    AttributeError: 'SuperMarioEnv' object has no attribute 'ale'
    """

    def __init__(self):
        pass

    def lives(self):
        # TODO(millman): Hook up to SimpleAiHandler.
        return 1


# From: nes/peripherals.py:323:
#   self.is_pressed[self.A] = int(state[self.A])
#   self.is_pressed[self.B] = int(state[self.B])
#   self.is_pressed[self.SELECT] = int(state[self.SELECT])
#   self.is_pressed[self.START] = int(state[self.START])
#   self.is_pressed[self.UP] = int(state[self.UP])
#   self.is_pressed[self.DOWN] = int(state[self.DOWN])
#   self.is_pressed[self.LEFT] = int(state[self.LEFT])
#   self.is_pressed[self.RIGHT] = int(state[self.RIGHT])
CONTROLLER_STATE_DESC = ["A", "B", "SELECT", "START", "UP", "DOWN", "LEFT", "RIGHT"]

def _describe_controller_vector(is_pressed: NdArrayUint8) -> str:
    pressed = [
        desc
        for is_button_pressed, desc in zip(is_pressed, CONTROLLER_STATE_DESC)
        if is_button_pressed
    ]
    return str(pressed)


def _to_controller_presses(buttons: list[str]) -> NdArrayUint8:
    is_pressed = np.zeros(8, dtype=np.uint8)
    for button in buttons:
        button_index = CONTROLLER_STATE_DESC.index(button.upper())
        is_pressed[button_index] = 1
    return is_pressed

class SimpleAiHandler:
    def __init__(self):
        self.frame_num = -1
        self.screen_buffer_image = np.zeros((3, 224, 224))

        self.last_reward_map = None
        self.reward_map = None
        self.reward_vector = None

        self.prev_info = None

    def shutdown(self):
        print("Shutting down ai handler")

    def update(self, frame: int, controller1, ram, screen_buffer_image):

        # Update rewards.
        self.last_reward_map = self.reward_map
        self.reward_map, self.reward_vector = compute_reward_map(
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

        _PRINT_FRAME_INFO = False

        # Print frame updates.
        if _PRINT_FRAME_INFO:
            always_changing_info = f"Frame: {frame:<5} Time left: {self.reward_map.time_left:<5} "
            controller_desc = _describe_controller_vector(controller1.is_pressed)
            new_info = f"Left pos: {self.reward_map.left_pos:<5} Reward: {self.reward_map} {self.reward_vector} Controller: {controller_desc}"
            if new_info == self.prev_info:
                # Clear out old line and display again.
                print(always_changing_info + new_info, end='\r', flush=True)
            else:
                # New info, start a new line.
                print('\n' + always_changing_info + new_info, end='\r', flush=True)

            self.prev_info = new_info

        # TODO(millman): Debugging for resetting episode.
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
            print("REWARD", self.reward_map, self.reward_vector)

            print("CONTROLLER", controller1.is_pressed)

        self.frame_num = frame
        self.screen_buffer_image = screen_buffer_image

        return True


# Reference: https://gymnasium.farama.org/introduction/create_custom_env/
class SuperMarioEnv(gym.Env):

    def __init__(self):
        self.action_controller_presses = [
            _to_controller_presses(['a']),
            _to_controller_presses(['b']),
            _to_controller_presses(['left']),
            _to_controller_presses(['right']),
            _to_controller_presses(['a', 'b']),
            _to_controller_presses(['a', 'left']),
            _to_controller_presses(['a', 'right']),
            _to_controller_presses(['b', 'left']),
            _to_controller_presses(['b', 'right']),
            _to_controller_presses(['a', 'b', 'left']),
            _to_controller_presses(['a', 'b', 'right']),
        ]

        self.action_space = gym.spaces.Discrete(len(self.action_controller_presses))

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

        self.ale = NesAle()

        # Initialize the NES Emulator.
        self.nes = NES(
            "./roms/Super_mario_brothers.nes",
            SimpleAiHandler(),
            sync_mode=SYNC_PYGAME,
            opengl=True,
            audio=False,
            verbose=False,
        )
        self.ai_handler = self.nes.ai_handler

    def _get_obs(self):
        # Get screen buffer.  Shape = (3, 244, 244)
        screen_buffer = np.array(self.ai_handler.screen_buffer_image)
        assert screen_buffer.dtype == np.uint8, f"Unexpected screen_buffer.dtype: {screen_buffer.dtype} != {np.uint8}"

        obs = screen_buffer

        return obs

    def _get_info(self):
        # TODO(millman): Put debug info here that shouldn't be used as game rewards.
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

        return observation, info

    def step(self, action_index: int):
        PRINT_CONTROLLER = False

        if PRINT_CONTROLLER:
            controller_desc = _describe_controller_vector(self.nes.controller1.is_pressed)
            print(f"Controller before: {controller_desc}")

        # Read the controller from the keyboard.
        if _USE_KEYBOARD_INPUT := True:
            self.nes.read_controller_presses()

        # If user pressed anything, avoid apply actions.
        if any(self.nes.controller1.is_pressed):
            if PRINT_CONTROLLER:
                controller_desc = _describe_controller_vector(self.nes.controller1.is_pressed)
                print(f"Controller (user pressed): {controller_desc}")
        else:
            # Convert an action_index into a specific set of controller actions.
            action = np.array(self.action_controller_presses[action_index])

            if False:
                # Set fixed action, for testing.
                # Right button.
                action[7] = 1

            if PRINT_CONTROLLER:
                controller_desc = _describe_controller_vector(action)
                print(f"Controller (input action): {controller_desc}")

            self.nes.controller1.set_state(action)

        if PRINT_CONTROLLER:
            controller_desc = _describe_controller_vector(self.nes.controller1.is_pressed)
            print(f"Controller after: {controller_desc}")

        # Take a step in the emulator.
        self.nes.run_frame()

        # Read off the current reward.  Convert to a single value reward for this timestep.
        reward = RewardMap.combine_reward_vector_single(self.ai_handler.reward_vector)

        # TODO(millman): set terminated/truncated based on lives and level change.
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
        # TODO(millman): All rendering is handled by pygame right now in the NES emulator; come back to this later.

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
