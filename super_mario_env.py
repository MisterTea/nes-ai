from typing import Literal
from typing import Optional

import gymnasium as gym
import numpy as np
import pygame
import pygame.freetype
import torch

from PIL import Image, ImageDraw
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


SCREEN_W = 240
SCREEN_H = 224

# Ram size shows up as 2048, but the max value in the RAM map is 34816, which is from NES RAM (2048) + Game (32768).
RAM_SIZE = 2048

class SimpleAiHandler:
    def __init__(self):
        self.frame_num = -1

        self.reset()

    def reset(self):
        self.ram = np.zeros(RAM_SIZE, dtype=np.uint8)
        self.screen_image = Image.fromarray(np.zeros((SCREEN_H, SCREEN_W, 3), dtype=np.uint8))

        self.last_reward_map = None
        self.reward_map, self.reward_vector = compute_reward_map(None, self.ram)

        self.prev_info = None

    def shutdown(self):
        print("Shutting down ai handler")

    def update(self, frame: int, controller1, ram: NdArrayUint8, screen_image: Image):
        assert screen_image.size == (SCREEN_W, SCREEN_H), f"Unexpected screen size: {screen_image.size} != {(SCREEN_W, SCREEN_H)}"

        if self.ram.shape != ram.shape:
            print(f"Unexpected ram size: {self.ram.shape} != {ram.shape}")

        self.ram = ram

        # Update rewards.
        self.last_reward_map = self.reward_map
        self.reward_map, self.reward_vector = compute_reward_map(
            self.last_reward_map, torch.from_numpy(ram).int()
        )

        _PRINT_FRAME_INFO = True

        # Print frame updates.
        if _PRINT_FRAME_INFO:
            if False:
                always_changing_info = f"Frame: {frame:<5} Time left: {self.reward_map.time_left:<5} "
                controller_desc = _describe_controller_vector(controller1.is_pressed)
                new_info = f"Left pos: {self.reward_map.left_pos:<5} Reward: {self.reward_map} {self.reward_vector} Controller: {controller_desc}"
                if new_info == self.prev_info:
                    # Clear out old line and display again.
                    print(always_changing_info + new_info, end='\r', flush=True)
                else:
                    # New info, start a new line.
                    print('\n' + always_changing_info + new_info, end='\r', flush=True)

            if False:
                # For debugging screen positions.

                always_changing_info = f"Frame: {frame:<5} Time left: {self.reward_map.time_left:<5} "

                # 0x0086	Player x position on screen
                # 0x006D	Player horizontal position in level

                # 0x00B5    Player vertical screen position
                # 0x00CE    Player y pos on screen (multiply with value at 0x00B5 to get level y pos)

                # Player horizontal position in level.
                level_x = ram[0x006D]

                # Screen X position.
                screen_x = ram[0x0086]

                # Player x pos within current screen offset.
                screen_offset = ram[0x03AD]

                new_info = f"level_x={level_x:>3} screen_x={screen_x:>3} screen_offset={screen_offset:>3}"

                if True: # new_info == self.prev_info:
                    # Clear out old line and display again.
                    print(always_changing_info + new_info, end='\r', flush=True)
                else:
                    # New info, start a new line.
                    print('\n' + always_changing_info + new_info, end='\r', flush=True)

                self.prev_info = new_info

            if False:
                # For debugging player state / end of level.

                # Player's state
                # 0x00 - Leftmost of screen
                # 0x01 - Climbing vine
                # 0x02 - Entering reversed-L pipe
                # 0x03 - Going down a pipe
                # 0x04 - Autowalk
                # 0x05 - Autowalk
                # 0x06 - Player dies
                # 0x07 - Entering area
                # 0x08 - Normal
                # 0x09 - Transforming from Small to Large (cannot move)
                # 0x0A - Transforming from Large to Small (cannot move)
                # 0x0B - Dying
                # 0x0C - Transforming to Fire Mario (cannot move)
                player_state = ram[0x000E]

                always_changing_info = f"Frame: {frame:<5} Time left: {self.reward_map.time_left:<5} "

                new_info = f"player_state={player_state}"

                if True: # new_info == self.prev_info:
                    # Clear out old line and display again.
                    print(always_changing_info + new_info, end='\r', flush=True)
                else:
                    # New info, start a new line.
                    print('\n' + always_changing_info + new_info, end='\r', flush=True)


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


def _run_until_not_dead(ai_handler, nes):
    ram = ai_handler.ram

    # Player's state
    # 0x00 - Leftmost of screen
    # 0x01 - Climbing vine
    # 0x02 - Entering reversed-L pipe
    # 0x03 - Going down a pipe
    # 0x04 - Autowalk
    # 0x05 - Autowalk
    # 0x06 - Player dies
    # 0x07 - Entering area
    # 0x08 - Normal
    # 0x09 - Transforming from Small to Large (cannot move)
    # 0x0A - Transforming from Large to Small (cannot move)
    # 0x0B - Dying
    # 0x0C - Transforming to Fire Mario (cannot move)
    player_state = ram[0x000E]

    # Wait for player dying animation to finish.
    while player_state == 0x0B:
        nes.run_frame()
        player_state = ai_handler.ram[0x000E]

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
    # for i in range(34):
    for i in range(50):
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
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 120}

    def __init__(self, render_mode: str | None = None, render_fps: int | None = None):
        self.render_mode = render_mode
        self.render_fps = render_fps

        # Screen setup.  2 Screens, 1 next to the other.
        self.screen_size = None
        self.multiscreen_image = Image.new("RGB", (SCREEN_W * 2, SCREEN_H))

        self.window = None
        self.clock = None

        self.action_controller_presses = [
            _to_controller_presses([]),
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

        self.observation_space = gym.spaces.Box(low=0, high=255, shape=(SCREEN_H, SCREEN_W, 3), dtype=np.uint8)

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

            # TOOD(millman): Testing out using screen from here.
            headless=True,
            show_hud=False,
        )
        self.ai_handler = self.nes.ai_handler

        # NOTE: reset() looks like it's called automatically by the gym environment, before starting.
        # self.reset()

        # For visualization.
        self.second_screen_image = Image.new(mode="RGB", size=(SCREEN_W, SCREEN_H))

    def _get_obs(self):
        # Get screen buffer.  Shape = (3, 240, 224)
        screen_image = np.asarray(self.ai_handler.screen_image)
        assert screen_image.shape == (SCREEN_H, SCREEN_W, 3), f"Unexpected screen_image.shape: {screen_image.shape} != {(SCREEN_H, SCREEN_W, 3)}"
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
        _run_until_not_dead(self.ai_handler, self.nes)

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

        if self.render_mode == "human":
            frame_image = self._build_frame()
            self._render_frame(frame_image)

        # Read off the current reward.  Convert to a single value reward for this timestep.
        reward = RewardMap.combine_reward_vector_single(self.ai_handler.reward_vector)

        self.ale._lives = self.ai_handler.reward_map.lives

        terminated = self.ale._lives <= 0
        truncated = False
        observation = self._get_obs()
        info = self._get_info()

        return observation, reward, terminated, truncated, info

    def render(self):
        if self.render_mode == "rgb_array":
            return self._build_frame()
        elif self.render_mode == "human":
            return None
        else:
            return None

    def _build_frame(self) -> NdArrayRGB8:
        # Use original image sizes here.  Scale everything later.
        w, h = SCREEN_W, SCREEN_H

        # Draw main screen.
        screen0 = self.ai_handler.screen_image
        assert screen0.size == (SCREEN_W, SCREEN_H), f"Unexpected screen0 size: {screen0.size} != {SCREEN_W, SCREEN_H}"
        self.multiscreen_image.paste(screen0, (0, 0))

        if False:
            values_image_np_uint8 = np.random.randint(0, 256, size=(SCREEN_H, SCREEN_W), dtype=np.uint8)
            values_gray = Image.fromarray(values_image_np_uint8, mode='L')
            values_rgb = values_gray.convert('RGB')

            self.second_screen_image = values_rgb

        # Draw debug screen.
        screen1 = self.second_screen_image
        assert screen1.size == (SCREEN_W, SCREEN_H), f"Unexpected screen1 size: {screen1.size} != {SCREEN_W, SCREEN_H}"
        self.multiscreen_image.paste(screen1, (w, 0))

        return self.multiscreen_image

    def _render_frame(self, frame_image: Image):
        # Scale up the image sizes.
        w, h = frame_image.size[0] * 3, frame_image.size[1] * 3

        last_rgb_array = frame_image.resize((w, h), resample=Image.Resampling.NEAREST)

        # Pygame is displayed as GRB?
        rgb_array = np.transpose(last_rgb_array, axes=(1, 0, 2))

        if self.screen_size is None:
            self.screen_size = rgb_array.shape[:2]

        assert (
            self.screen_size == rgb_array.shape[:2]
        ), f"The shape of the rgb array has changed from {self.screen_size} to {rgb_array.shape[:2]}"

        if self.window is None:
            pygame.init()
            pygame.display.init()
            self.window = pygame.display.set_mode(self.screen_size)

        if self.clock is None:
            self.clock = pygame.time.Clock()

        surf = pygame.surfarray.make_surface(rgb_array)
        self.window.blit(surf, (0, 0))
        pygame.event.pump()

        if self.render_fps:
            print(f"TICKING TO {self.render_fps}")
            self.clock.tick(self.render_fps)

        pygame.display.flip()

    def close(self):
        if self.window is not None:
            pygame.display.quit()
            pygame.quit()
        super().close()
