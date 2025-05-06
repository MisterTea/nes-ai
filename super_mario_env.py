import random
import time
from collections import deque
from typing import Any, Literal, Optional

import gymnasium as gym
import numpy as np
import pygame
import torch

from PIL import Image
from nes import NES, SYNC_NONE, SYNC_PYGAME
from nes_ai.ai.base import RewardIndex, RewardMap, compute_reward_map

from super_mario_env_ram_hacks import skip_after_step, life

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

CONTROLLER_NOOP = _to_controller_presses([])


# Action spaces adapted from:
# https://github.com/Kautenja/gym-super-mario-bros/blob/master/gym_super_mario_bros/actions.py
# venv/lib/python3.11/site-packages/gym_super_mario_bros/actions.py

# actions for the simple run right environment
RIGHT_ONLY = [
    [],
    ['right'],
    ['right', 'a'],
    ['right', 'b'],
    ['right', 'a', 'b'],
]


# actions for very simple movement
SIMPLE_MOVEMENT = [
    [],
    ['right'],
    ['right', 'a'],
    ['right', 'b'],
    ['right', 'a', 'b'],
    ['a'],
    ['left'],
]


# actions for more complex movement
COMPLEX_MOVEMENT = [
    [],
    ['right'],
    ['right', 'a'],
    ['right', 'b'],
    ['right', 'a', 'b'],
    ['a'],
    ['left'],
    ['left', 'a'],
    ['left', 'b'],
    ['left', 'a', 'b'],
    ['down'],
    ['up'],
]

COMPLEX_LEFT_RIGHT = [
    [],
    ['right'],
    ['right', 'a'],
    ['right', 'b'],
    ['right', 'a', 'b'],
    ['a'],
    ['left'],
    ['left', 'a'],
    ['left', 'b'],
    ['left', 'a', 'b'],
]


SCREEN_W = 240
SCREEN_H = 224

# Ram size shows up as 2048, but the max value in the RAM map is 34816, which is from NES RAM (2048) + Game (32768).
RAM_SIZE = 2048

class SimpleAiHandler:
    def __init__(self):
        self.frame_num = -1

        self.reset()

    def reset(self):
        self.last_reward_map = None

        ram = np.zeros(RAM_SIZE, dtype=np.uint8)
        self.reward_map, self.reward_vector = compute_reward_map(None, ram)

        self.prev_info = None

    def shutdown(self):
        print("Shutting down ai handler")

    def update(self, frame: int, controller1: NdArrayUint8, ram: NdArrayUint8, screen_image: Image):
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

        return True


class SimpleScreenNx1:
    def __init__(self, screen_size: tuple[int, int], scale: int, n: int):
        self.scale = scale

        w, h = screen_size
        self.screen_size = screen_size
        self.screen_size_scaled = (w * scale, h * scale)
        self.window_size = (self.screen_size[0] * n, self.screen_size[1])

        self.window = None

        self.combined_surf = pygame.Surface((w * n, h))
        self.surfs = [
            # NOTE: Rect(left, top, width, height)
            self.combined_surf.subsurface(pygame.Rect(w*i, 0, w, h))
            for i in range(n)
        ]
        self.combined_surf_scaled = pygame.Surface(self.window_size)

    def get_as_image(self, screen_index: int) -> Image:
        surf = self.surfs[screen_index]

        data = pygame.image.tostring(surf, 'RGB')
        width, height = surf.get_size()
        image = Image.frombytes(mode='RGB', size=(width, height), data=data)

        return image

    def blit_image_np(self, image_np: NdArrayRGB8, screen_index: int):
        # print(f"SURF SIZE: {self.surfs[screen_index].get_size()}  image.shape={image_np.shape}")
        pygame.surfarray.blit_array(self.surfs[screen_index], image_np)

    def blit_image(self, image: Image, screen_index: int):
        assert image.mode == 'RGB', f"Unexpected image mode: {image.mode} != RGB"
        image_np = np.asarray(image).swapaxes(0, 1)
        pygame.surfarray.blit_array(self.surfs[screen_index], image_np)

    def show(self):
        if self.window is None:
            pygame.init()
            pygame.display.init()
            self.window = pygame.display.set_mode(self.window_size)

        if self.scale != 1:
            pygame.transform.scale(surface=self.combined_surf, size=self.window_size, dest_surface=self.combined_surf_scaled)
            self.window.blit(self.combined_surf_scaled, dest=(0, 0))
        else:
            pygame.transform.scale(surface=self.combined_surf, size=self.window_size, dest_surface=self.combined_surf_scaled)
            self.window.blit(self.combined_surf, dest=(0, 0))

        pygame.event.pump()
        pygame.display.flip()

    def close(self):
        if self.window is not None:
            pygame.display.quit()
            pygame.quit()


_DEBUG_LEVEL_START = False

def _debug_level_from_ram(ram: NdArrayUint8, frame_num: int, desc: str):
    if not _DEBUG_LEVEL_START:
        return

    level = ram[0x0760]
    game_mode = ram[0x0770]
    prelevel = ram[0x075E]
    prelevel_timer = ram[0x07A0]
    level_entry = ram[0x0752]
    before_level_load = ram[0x0753]
    level_loading = ram[0x0772]
    print(f"{desc}: frame={frame_num} {level=} {game_mode=} {prelevel=} {prelevel_timer=} {level_entry=} {before_level_load=} {level_loading=}")


def _skip_start_screen(nes: Any):
    ram = nes.ram()

    _debug_level_from_ram(ram, frame_num=nes.get_frame_num(), desc="RUNNING GAME START")

    # Run frames until start screen.
    # TODO(millman): Not sure why this is 34 frames?  Found by binary searching until this worked.
    # for i in range(34):
    for i in range(34 * 4):
        nes.run_frame()

    _debug_level_from_ram(ram, frame_num=nes.get_frame_num(), desc="BEFORE START PRESSED")

    # Press start.
    nes.controller1.set_state(_to_controller_presses(['start']))

    # Run one frame.
    nes.run_frame()

    # Make sure "game mode" is set to "normal".
    game_mode = ram[0x0770]
    assert game_mode == 1, f"Unexpected game mode (0=demo 1=normal): {game_mode} != 1"

    # Set controller back to no-op.
    nes.controller1.set_state(_to_controller_presses([]))

    _debug_level_from_ram(ram, frame_num=nes.get_frame_num(), desc="AFTER START PRESSED")

    # We're now ready to play.
    _debug_level_from_ram(ram, frame_num=nes.get_frame_num(), desc="READY TO PLAY")


def _left_pos(ram: NdArrayUint8) -> int:
    return (int(ram[0x006D]) * 256) + int(ram[0x0086])


# Reference: https://gymnasium.farama.org/introduction/create_custom_env/
class SuperMarioEnv(gym.Env):
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 120}

    def __init__(self, render_mode: str | None = None, render_fps: int | None = None, reset_to_save_state: bool = False):
        self.resets = 0

        self.reset_to_save_state = reset_to_save_state

        # Save state buffer.
        self.save_states_time_buffer = deque(maxlen=5)
        self.save_states_distance_buffer = deque(maxlen=5)
        self.seconds_between_saves = 5.0
        self.left_pos_between_saves = 100
        self.last_save_time = time.time()
        self.last_save_left_pos = 0

        self.render_mode = render_mode
        self.render_fps = render_fps

        # Screen setup.  2 Screens, 1 next to the other.
        self.screen = SimpleScreenNx1((SCREEN_W, SCREEN_H), scale=3, n=5)

        self.clock = None

        self.action_controller_presses = [
            _to_controller_presses(buttons)
            for buttons in SIMPLE_MOVEMENT
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

        self.observation_space = gym.spaces.Box(low=0, high=255, shape=(SCREEN_W, SCREEN_H, 3), dtype=np.uint8)

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

        # Initialize so we can run past the start screen.
        self.nes.reset()
        self.nes.run_init()

        # Skip start screen.
        _skip_start_screen(self.nes)

        # Save a snapshot to restore on next calls to reset.
        self.start_state = self.nes.save()

        # NOTE: the ai_handler has not run yet, so we need to pull the number of lives right out
        #  of memory.
        self.ale._lives = life(self.nes.ram())
        self.ai_handler.update(frame=0, controller1=_to_controller_presses([]), ram=self.nes.ram(), screen_image=None)
        assert self.ai_handler.reward_map.lives == self.ale.lives(), f"Mismatched lives: reward_map={self.ai_handler.reward_map.lives} ram={self._ale.lives()}"

    def _get_obs(self) -> NdArrayRGB8:
        screen_view = self.nes.get_screen_view()
        screen_view_np = self._screen_view_to_np(screen_view)

        assert screen_view_np.shape == (SCREEN_W, SCREEN_H, 3), f"Unexpected screen_view_np.shape: {screen_view_np.shape} != {(SCREEN_W, SCREEN_H, 3)}"
        assert screen_view_np.dtype == np.uint8, f"Unexpected screen_view_np.dtype: {screen_view_np.dtype} != {np.uint8}"

        return screen_view_np

    def _get_info(self):
        # TODO(millman): Put debug info here that shouldn't be used as game rewards.
        # return {
        #     "distance": np.linalg.norm(
        #         self._agent_location - self._target_location, ord=1
        #     )
        # }
        return {}

    def reset(self, seed: Optional[int] = None, options: Optional[dict] = None):
        self.resets += 1

        # TODO(millman): fix seed, etc.

        # We need the following line to seed self.np_random
        # super().reset(seed=seed)

        # Get left_pos before resetting.
        left_pos = _left_pos(self.nes.ram())

        # Reset CPU and controller.
        self.nes.reset()
        self.nes.controller1.set_state(_to_controller_presses([]))

        if self.reset_to_save_state and (self.save_states_time_buffer or self.save_states_distance_buffer):
            num_states = len(self.save_states_time_buffer) + len(self.save_states_distance_buffer)

            if _ALWAYS_INCLUDE_START_STATE := False:
                # Always include some chance of doing a regular reset.
                num_states += 1

            state_i = random.randint(0, num_states)

            if state_i < len(self.save_states_time_buffer):
                state, state_time = self.save_states_time_buffer[state_i]
                self.nes.load(state)
                print(f"Reset to state: {time.time() - state_time:.2f}s ago")
            elif state_i < len(self.save_states_time_buffer) + len(self.save_states_distance_buffer):
                i = state_i - len(self.save_states_time_buffer)
                state, state_left_pos = self.save_states_distance_buffer[i]
                self.nes.load(state)
                print(f"Reset to state: {int(left_pos) - state_left_pos} distance to the left")
            else:
                self.nes.load(self.start_state)
                print("Reset to starting state")

        else:
            # Load from saved state, after start screen.
            self.nes.load(self.start_state)

        # Get initial values.
        observation = self._get_obs()
        info = self._get_info()

        # Reset ai handler.
        self.ai_handler.update(frame=0, controller1=_to_controller_presses([]), ram=self.nes.ram(), screen_image=None)
        self.ale._lives = life(self.nes.ram())
        assert self.ai_handler.reward_map.lives == self.ale.lives(), f"Mismatched lives: reward_map={self.ai_handler.reward_map.lives} ram={self._ale.lives()}"

        self.last_observation = observation

        return observation, info

    def step(self, action_index: int):
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

        self.ale._lives = life(self.nes.ram())
        assert self.ai_handler.reward_map.lives == self.ale.lives(), f"Mismatched lives: reward_map={self.ai_handler.reward_map.lives} ram={self._ale.lives()}"

        terminated = self.ale._lives <= 0
        truncated = False
        observation = self._get_obs()
        info = self._get_info()

        self.last_observation = observation

        # Show the screen, if requested.
        if self.render_mode == "human":
            self.screen.blit_image_np(observation, screen_index=0)
            self.screen.show()

        if self.reset_to_save_state and not terminated:
            # If it's been a few seconds since we last saved, create a new save state.
            now = time.time()
            if now - self.last_save_time > self.seconds_between_saves:
                self.save_states_time_buffer.append((self.nes.save(), now))
                self.last_save_time = time.time()

            # Reset position if we're in a new level or new world.
            if self.ai_handler.reward_vector[RewardIndex.WORLD] or self.ai_handler.reward_vector[RewardIndex.LEVEL]:
                self.last_save_left_pos = 0

            # If we've moved far enough, create a new save state.
            left_pos = _left_pos(self.nes.ram())
            if left_pos - self.last_save_left_pos > self.left_pos_between_saves:
                self.save_states_distance_buffer.append((self.nes.save(), int(left_pos)))
                self.last_save_left_pos = left_pos

        # Speed through any prelevel screens, dying animations, etc. that we don't care about.
        skip_after_step(self.nes)

        return observation, reward, terminated, truncated, info

    def render(self):
        if self.render_mode == "rgb_array":
            return self.last_observation
        elif self.render_mode == "human":
            self.screen.show()
            return None
        else:
            return None

    @staticmethod
    def _screen_view_to_np(screen_view: Any) -> NdArrayRGB8:
        assert screen_view.shape == (240, 224), f"Unexpected screen_view shape: {screen_view.shape} != {(240, 224)}"

        # NOTE: These operations are carefully constructed to avoid memory copies, they are all views.
        #   Starting type is: (240, 224) uint32 as BGRA.
        #   Ending type is: (240, 224, 3) uint8 as RGB.

        screen_view_np = np.asarray(screen_view, copy=False)
        screen_view_bgra = screen_view_np.view(np.uint8).reshape((240, 224, 4))
        screen_view_bgr = screen_view_bgra[:, :, :3]
        screen_view_rgb = screen_view_bgr[:, :, ::-1]

        if False:
            def _is_copy(arr):
                return 'view' if arr.base is not None else 'new'

            print()
            print(f"SCREEN VIEW: type={type(screen_view)} shape={screen_view.shape} size={screen_view.size} base={_is_copy(screen_view)}")
            print(f"SCREEN VIEW NP: shape={screen_view_np.shape} size={screen_view_np.size} cont={screen_view_np.flags['C_CONTIGUOUS']} base={_is_copy(screen_view_np)}")
            print(f"SCREEN VIEW BGRA: shape={screen_view_bgra.shape} size={screen_view_bgra.size} cont={screen_view_bgra.flags['C_CONTIGUOUS']} base={_is_copy(screen_view_bgra)}")
            print(f"SCREEN VIEW BGR: shape={screen_view_bgr.shape} size={screen_view_bgr.size} cont={screen_view_bgr.flags['C_CONTIGUOUS']} base={_is_copy(screen_view_bgr)}")
            print(f"SCREEN VIEW RGB: shape={screen_view_rgb.shape} size={screen_view_rgb.size} cont={screen_view_rgb.flags['C_CONTIGUOUS']} base={_is_copy(screen_view_rgb)}")

        return screen_view_rgb

    def close(self):
        self.screen.close()
        super().close()
