from typing import Literal
from typing import Optional

import gymnasium as gym
import numpy as np
import torch

from PIL import Image
from nes import NES, SYNC_NONE, SYNC_PYGAME
from nes_ai.ai.base import RewardMap, compute_reward_map

NdArrayUint8 = np.ndarray[np.dtype[np.uint8]]
NdArrayRGB8 = np.ndarray[tuple[Literal[4]], np.dtype[np.uint8]]

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
        self._lives = 1

    def lives(self):
        return self._lives


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


# Ram size shows up as 34816, which is from NES RAM (2048) + Game (32768).
RAM_SIZE = 32768 + 2048

class SimpleAiHandler:
    def __init__(self):
        self.frame_num = -1

        self.reset()

    def reset(self):
        self.ram = np.zeros(RAM_SIZE, dtype=np.uint8)
        self.screen_image = Image.fromarray(np.zeros((224, 224, 3), dtype=np.uint8))

        self.last_reward_map = None
        self.reward_map, self.reward_vector = compute_reward_map(None, self.ram)

        self.prev_info = None

    def shutdown(self):
        print("Shutting down ai handler")

    def update(self, frame: int, controller1, ram: NdArrayUint8, screen_image: Image):
        assert screen_image.size == (224, 224)

        self.ram = ram

        # Update rewards.
        self.last_reward_map = self.reward_map
        self.reward_map, self.reward_vector = compute_reward_map(
            self.last_reward_map, torch.from_numpy(ram).int()
        )

        _PRINT_FRAME_INFO = True

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

        self.frame_num = frame
        self.screen_image = screen_image

        return True


_DEBUG_LEVEL_START = False

def _debug_level_from_ram(ai_handler, desc: str):
    if not _DEBUG_LEVEL_START:
        return

    ram = ai_handler.ram

    level = ram[0x0760]
    game_mode = ram[0x0770]
    prelevel = ram[0x075E]
    prelevel_timer = ram[0x07A0]
    level_entry = ram[0x0752]
    before_level_load = ram[0x0753]
    level_loading = ram[0x0772]
    print(f"{desc}: frame={ai_handler.frame_num} {level=} {game_mode=} {prelevel=} {prelevel_timer=} {level_entry=} {before_level_load=} {level_loading=}")


def _run_until_level_started(ai_handler, nes):
    # Wait until the level is set.
    # The level loading byte looks like it counts from 1 (during load) to 2 (right before load) to 3 (ready).
    level_loading = ai_handler.ram[0x0772]

    if level_loading >= 3:
        # We're already in a level, don't do anything.
        return
    else:
        # We're waiting for a level to load.  Run until load, and then notify.
        while level_loading <= 2:
            _debug_level_from_ram(ai_handler, "WAITING FOR LEVEL")
            level_loading = ai_handler.ram[0x0772]

            nes.run_frame()


def _run_until_game_started(ai_handler, nes):
    _debug_level_from_ram(ai_handler, "RUNNING GAME START")

    # Run frames until start screen.
    # TODO(millman): Not sure why this is 34 frames?  Found by binary searching until this worked.
    for i in range(34):
        nes.run_frame()

    _debug_level_from_ram(ai_handler, "BEFORE START PRESSED")

    # Press start.
    nes.controller1.set_state(_to_controller_presses(['start']))

    # Run one frame.
    nes.run_frame()

    # Make sure "game mode" is set to "normal".
    game_mode = ai_handler.ram[0x0770]
    assert game_mode == 1, f"Unexpected game mode (0=demo 1=normal): {game_mode} != 1"

    # Set controller back to no-op.
    nes.controller1.set_state(_to_controller_presses([]))

    _debug_level_from_ram(ai_handler, "AFTER START PRESSED")

    _run_until_level_started(ai_handler, nes)

    # We're now ready to play.
    _debug_level_from_ram(ai_handler, "READY TO PLAY")




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
            # sync_mode=SYNC_PYGAME,
            sync_mode=SYNC_NONE,
            opengl=True,
            audio=False,
            verbose=False,
        )
        self.ai_handler = self.nes.ai_handler

        # NOTE: reset() looks like it's called automatically by the gym environment, before starting.
        # self.reset()

    def _get_obs(self):
        # Get screen buffer.  Shape = (3, 244, 244)
        screen_image = np.array(self.ai_handler.screen_image)
        assert screen_image.dtype == np.uint8, f"Unexpected screen_image.dtype: {screen_image.dtype} != {np.uint8}"

        obs = screen_image

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
        # TODO(millman): fix seed, etc.

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

        # Reset ai handler.
        self.ai_handler.reset()

        # Reset CPU and controller.
        self.nes.reset()
        self.nes.run_init()

        # Step until game started.
        _run_until_game_started(self.ai_handler, self.nes)

        # Get initial values.
        observation = self._get_obs()
        info = self._get_info()

        self.ale._lives = self.ai_handler.reward_map.lives

        return observation, info

    def step(self, action_index: int):
        # Wait for level to start.  Does nothing if we're already in a level.
        # Necessary to handle continuing episodes after losing a life.
        _run_until_level_started(self.ai_handler, self.nes)

        PRINT_CONTROLLER = False

        if PRINT_CONTROLLER:
            controller_desc = _describe_controller_vector(self.nes.controller1.is_pressed)
            print(f"Controller before: {controller_desc}")

        # Read the controller from the keyboard.
        if _USE_KEYBOARD_INPUT := True:
            self.nes.read_controller_presses()

        # If user pressed anything, avoid applying actions.
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

        self.ale._lives = self.ai_handler.reward_map.lives

        # TODO(millman): set terminated/truncated based on lives and level change.
        terminated = self.ale._lives <= 0
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
