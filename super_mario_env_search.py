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
from nes_ai.ai.base import RewardIndex, RewardMap, compute_reward_map, get_level, get_time_left, get_world

from super_mario_env import _skip_start_screen, _describe_controller_vector, _to_controller_presses
from super_mario_env_ram_hacks import _skip_change_area, _skip_occupied_states, skip_after_step, life

NdArrayUint8 = np.ndarray[np.dtype[np.uint8]]
NdArrayRGB8 = np.ndarray[tuple[Literal[4]], np.dtype[np.uint8]]


SCREEN_W = 240
SCREEN_H = 224

# Ram size shows up as 2048, but the max value in the RAM map is 34816, which is from NES RAM (2048) + Game (32768).
RAM_SIZE = 2048

class SimpleAiHandler:
    def __init__(self):
        self.frame_num = -1

    def reset(self):
        pass

    def shutdown(self):
        pass

    def update(self, frame: int, controller1: NdArrayUint8, ram: NdArrayUint8, screen_image: Image):
        self.frame_num = frame
        return True


class SimpleScreenRxC:
    def __init__(self, screen_size: tuple[int, int], scale: int, rows: int, cols: int):
        r, c = rows, cols
        self.rows = rows
        self.cols = cols
        self.scale = scale

        w, h = screen_size
        self.screen_size = screen_size
        self.screen_size_scaled = (w * scale, h * scale)
        self.window_size = (self.screen_size[0] * c, self.screen_size[1] * r)

        self.window = None

        self.combined_surf = pygame.Surface((w * c, h * r))
        self.surfs = [
            # NOTE: Rect(left, top, width, height)
            self.combined_surf.subsurface(pygame.Rect(w*ci, h*ri, w, h))
            for ri in range(rows)
            for ci in range(cols)
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


def get_x_pos(ram: NdArrayUint8) -> int:
    return (int(ram[0x006D]) * 256) + int(ram[0x0086])


def get_y_pos(ram: NdArrayUint8) -> int:
    return ram[0x00CE]


# Reference: https://gymnasium.farama.org/introduction/create_custom_env/
class SuperMarioEnv(gym.Env):
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 120}

    def __init__(self,
        render_mode: str | None = None,
        render_fps: int | None = None,
        screen_rc: tuple[int, int] = (1, 1),
        world_level: tuple[int, int] | None = None,
    ):
        self.resets = 0

        self.render_mode = render_mode
        self.render_fps = render_fps

        # Screen setup.  2 Screens, 1 next to the other.
        self.screen = SimpleScreenRxC((SCREEN_W, SCREEN_H), scale=1, rows=screen_rc[0], cols=screen_rc[1])

        self.clock = None

        self.action_space = gym.spaces.MultiBinary(8)

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

        # Initialize the NES Emulator.
        self.nes = NES(
            "./roms/Super_mario_brothers.nes",
            SimpleAiHandler(),
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
        _skip_start_screen(self.nes, world_level=world_level)

        # Save a snapshot to restore on next calls to reset.
        self.start_state = self.nes.save()

    def _get_obs(self) -> NdArrayRGB8:
        screen_view = self.nes.get_screen_view()
        screen_view_np = self._screen_view_to_np(screen_view)

        assert screen_view_np.shape == (SCREEN_W, SCREEN_H, 3), f"Unexpected screen_view_np.shape: {screen_view_np.shape} != {(SCREEN_W, SCREEN_H, 3)}"
        assert screen_view_np.dtype == np.uint8, f"Unexpected screen_view_np.dtype: {screen_view_np.dtype} != {np.uint8}"

        return screen_view_np

    def _get_info(self):
        return {}

    def reset(self, seed: Optional[int] = None, options: Optional[dict] = None):
        self.resets += 1

        # TODO(millman): fix seed, etc.

        # We need the following line to seed self.np_random
        # super().reset(seed=seed)

        # Reset CPU and controller.
        self.nes.reset()
        self.nes.controller1.set_state(_to_controller_presses([]))

        # Load from saved state, after start screen.
        self.nes.load(self.start_state)

        # Get initial values.
        observation = self._get_obs()
        info = self._get_info()

        self.last_observation = observation

        self.ai_handler.reset()

        return observation, info

    def step(self, controller_presses: NdArrayUint8):
        ram = self.nes.ram()
        prev_lives = life(ram)

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
            action = controller_presses

            if False:
                # Set fixed action, for testing.
                # Right button.
                action[7] = 1

            if PRINT_CONTROLLER:
                print(f"CONTROLLER 1 ACTION: {action}")
                controller_desc = _describe_controller_vector(action)
                print(f"Controller (input action): {controller_desc}")

            self.nes.controller1.set_state(action)

        if PRINT_CONTROLLER:
            controller_desc = _describe_controller_vector(self.nes.controller1.is_pressed)
            print(f"Controller after: {controller_desc}")

        # Take a step in the emulator.
        self.nes.run_frame()

        # Read off the current reward.  Convert to a single value reward for this timestep.
        reward = 0

        lives = life(self.nes.ram())
        delta_lives = lives - prev_lives

        terminated = delta_lives < 0 or delta_lives > 1
        truncated = False
        observation = self._get_obs()
        info = self._get_info()

        self.last_observation = observation

        # Show the screen, if requested.
        if self.render_mode == "human":
            self.screen.blit_image_np(observation, screen_index=0)
            self.screen.show()

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
