from typing import Any, Literal, Optional

import gymnasium as gym
import matplotlib.pyplot as plt
import numpy as np
import pygame

from PIL import Image

NdArrayUint8 = np.ndarray[np.dtype[np.uint8]]
NdArrayRGB8 = np.ndarray[tuple[Literal[4]], np.dtype[np.uint8]]


# Action spaces adapted from:
# https://github.com/Kautenja/gym-super-mario-bros/blob/master/gym_super_mario_bros/actions.py
# venv/lib/python3.11/site-packages/gym_super_mario_bros/actions.py

# actions for the simple run right environment
SIMPLE_MOVEMENT = [
    'left',
    'right',
    'up',
    'down',
]

MOVEMENT_TO_OFFSET_RC = {
    'noop': (0, 0),
    'left': (0, -1),
    'right': (0, 1),
    'up': (1, 0),
    'down': (-1, 0),
    'left-up': (1, -1),
    'right-up': (1, 1),
    'left-down': (-1, -1),
    'right-down': (-1, 1),
}


class SimpleScreenRxC:
    def __init__(self, screen_size: tuple[int, int], scale: int, rows: int, cols: int):
        r, c = rows, cols
        self.rows = rows
        self.cols = cols
        self.scale = scale

        w, h = screen_size
        self.screen_size = screen_size
        self.screen_size_scaled = (w * scale, h * scale)
        self.window_size = (self.screen_size_scaled[0] * c, self.screen_size_scaled[1] * r)

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
        image_np = image_np.swapaxes(0, 1)
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



SCREEN_W = 51
SCREEN_H = 51

EMPTY = 0
WALL = 1

OBS_W, OBS_H = 84, 84



# Reference: https://gymnasium.farama.org/introduction/create_custom_env/
class FourRoomsEnv(gym.Env):
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 120}

    def __init__(self,
        render_mode: str | None = None,
        render_fps: int | None = None,
        screen_rc: tuple[int, int] = (1, 1),
    ):
        self.render_mode = render_mode

        # self.action_space = gym.spaces.Discrete(len(SIMPLE_MOVEMENT))
        self.action_space = gym.spaces.Discrete(len(SIMPLE_MOVEMENT))

        self.observation_space = gym.spaces.Box(low=0, high=255, shape=(OBS_W, OBS_H), dtype=np.uint8)
        # self.observation_space = gym.spaces.Box(low=0, high=255, shape=(84, 84), dtype=np.uint8)
        #self.observation_space = gym.spaces.Box(low=0, high=255, shape=(SCREEN_W, SCREEN_H, 3), dtype=np.uint8)
        #self.observation_space = gym.spaces.MultiDiscrete([SCREEN_H, SCREEN_W]) # Box(low=0, high=255, shape=(SCREEN_W, SCREEN_H, 3), dtype=np.uint8)

        self.screen = SimpleScreenRxC((SCREEN_W, SCREEN_H), scale=5, rows=screen_rc[0], cols=screen_rc[1])

        # Build action mapping.
        self._action_index_to_grid_offset = {
            i: MOVEMENT_TO_OFFSET_RC[action_str]
            for i, action_str in enumerate(SIMPLE_MOVEMENT)
        }

        self.viridis = plt.get_cmap('viridis')

        # Initialize room.
        # Grid of 51x51:
        #   * 4 rooms, top-left, top-right, bottom-left, bottom-right.
        #   * Each room is 25x25.
        #   * A wall separates each room.
        #   * Each wall has a single gridspace passage through it.

        # Initialize grid to empty.
        grid = np.zeros((SCREEN_H, SCREEN_W), dtype=np.int64)

        # Fill in interior center-vertical wall.
        grid[:, 25] = WALL

        # Fill in interior center-horizontal wall.
        grid[25, :] = WALL

        # Add openings in walls:
        #   - middle-left
        #   - middle-right
        #   - middle-top
        #   - middle-bottom
        offsets = [
            [-5,   0],
            [ 5,   0],
            [ 0, -10],
            [ 0,  10],
        ]

        for r, c in offsets:
            grid[25 + r, 25 + c] = EMPTY

        self._grid_walls = grid
        self._start_pos = np.array((0, 50))
        self._goal_pos = np.array((50, 0))

        # Agent init.
        self._grid_counts = np.zeros((51, 51), dtype=np.int64)
        self._agent_pos = self._start_pos

        if False:
            print("GRID:")
            for j, row in enumerate(self._grid_walls):
                if j > 0:
                    print()

                for i, is_wall in enumerate(row):
                    if i > 0:
                        print(" ", end="")
                    print(is_wall, end="")

                print()

        if True:
            print("GRID POSITION AT EDGE")
            for (r, c) in [(0, 0), (0, 50), (50, 0), (50, 50)]:
                print(f"  {r},{c}: {self._grid_walls[r,c]}")



    def get_debug_obs(self) -> NdArrayRGB8:
        # Normalize grid counts to range (0, 255).
        grid_f = self._grid_counts.astype(np.float32)
        grid_rgba_f = self.viridis(grid_f / grid_f.max())
        grid_rgb = (grid_rgba_f[..., :3] * 255).astype(np.uint8)

        # Set unexplored spaces as white.
        # grid_rgb[self._grid_counts == 0] = 255

        # Set unexplored spaces as black.
        grid_rgb[self._grid_counts == 0] = 0

        # Goal in blue.
        goal_r, goal_c = self._goal_pos
        grid_rgb[goal_r, goal_c, :] = (0, 0, 255)

        # DEBUG POSITIONS IN BLUE
        if False:
            grid_rgb[0, 0] = (0, 0, 255)
            grid_rgb[0, 50] = (0, 0, 255)
            grid_rgb[50, 0] = (0, 0, 255)
            grid_rgb[50, 50] = (0, 0, 255)

        # Repeat for RGB.
        screen_view_np = grid_rgb
        # screen_view_np = np.stack([grid_gray]*3, axis=-1)
        # screen_view_np = np.repeat(grid_gray[..., np.newaxis], 3, axis=-1)

        # Draw walls as green.
        if False:
            for r, row in enumerate(self._grid_walls):
                for c, is_wall in enumerate(row):
                    if is_wall:
                        screen_view_np[r, c, :] = [0, 255, 0]

            print(f"SCREEN VIEW SHAPE: {screen_view_np.shape}")

        if True:
            screen_view_np[self._grid_walls == WALL] = [0, 255, 0]

        # Draw agent as red.
        agent_r, agent_c = self._agent_pos
        screen_view_np[agent_r, agent_c, :] = [255, 0, 0]

        # Convert grid counts to an image.
        assert screen_view_np.shape == (SCREEN_W, SCREEN_H, 3), f"Unexpected screen_view_np.shape: {screen_view_np.shape} != {(SCREEN_W, SCREEN_H, 3)}"
        assert screen_view_np.dtype == np.uint8, f"Unexpected screen_view_np.dtype: {screen_view_np.dtype} != {np.uint8}"

        return screen_view_np

    def _get_obs(self) -> NdArrayRGB8:
        # Get a grayscale grid around the agent position.
        #obs = np.zeros((OBS_W, OBS_H), dtype=np.uint8)
        obs = np.zeros((84, 84), dtype=np.uint8)

        for j in range(-3, 3):
            for i in range(-3, 3):
                r = self._agent_pos[0] + j
                c = self._agent_pos[1] + i
                if r < 0 or r >= SCREEN_H or c < 0 or c >= SCREEN_W:
                    continue

                #obs[j+3, i+3] = self._grid_walls[r, c] * 255
                obs[j+42, i+42] = self._grid_walls[r, c] * 255

        # Repeat channels.
        # obs_rgb = np.stack([obs]*3, axis=-1).transpose((2, 0, 1))
        # return obs_rgb

        assert not np.isnan(obs).any(), "Observation has NaNs!"

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
        # Reset to starting point, top right.
        self._grid_counts = np.zeros((SCREEN_H, SCREEN_W), dtype=np.int64)
        self._agent_pos = self._start_pos

        # Get initial values.
        observation = self._get_obs()
        info = self._get_info()

        self.last_observation = observation

        return observation, info

    def step(self, action_index: int):
        # print(f"action_index: {action_index}")
        # action_index = np.random.randint(0, 4)

        # Move position according to 4 rooms.
        dr, dc = self._action_index_to_grid_offset[action_index]

        new_rc = self._agent_pos + np.array((dr, dc), dtype=np.int64)
        new_r, new_c = new_rc

        if False:
            if (self._agent_pos==0).any() or (self._agent_pos==1).any():
                print(f"NEAR EDGE: {self._agent_pos} + {dr},{dc} -> {new_r},{new_c}")

        if new_r < 0 or new_r >= SCREEN_H or new_c < 0 or new_c >= SCREEN_W:
            # Out of bounds, don't move.
            pass
        elif self._grid_walls[new_r, new_c] == WALL:
            # Hit a wall, don't move.
            pass
        else:
            # Empty space, move agent.
            self._agent_pos = new_rc

            # Update grid counts.
            self._grid_counts[new_r, new_c] += 1

        # Get reward in the new position.
        if (self._agent_pos == self._goal_pos).all():
            at_goal = True
        else:
            at_goal = False

        reward = 0.0 if at_goal else -1.0

        # print(f"REWARD: {reward}")

        if False:
            if (self._agent_pos==0).any() or (self._agent_pos==1).any():
                r, c = self._agent_pos
                print(f"GRID COUNTS AT EDGE: {r},{c}: {self._grid_counts[r, c]}")


        terminated = at_goal
        truncated = False
        observation = self._get_obs()
        info = self._get_info()

        # Show the screen, if requested.
        if self.render_mode == "human":
            human_obs = self.get_debug_obs()
            self.screen.blit_image_np(human_obs, screen_index=0)
            self.screen.show()

        return observation, reward, terminated, truncated, info

    def render(self):
        if self.render_mode == "rgb_array":
            return self.last_observation
        elif self.render_mode == "human":
            self.screen.show()
            return None
        else:
            return None

    def close(self):
        self.screen.close()
        super().close()
