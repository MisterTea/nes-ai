from typing import Any, Literal, Optional

import numpy as np
import torch
from PIL import Image

NdArrayUint8 = np.ndarray[np.dtype[np.uint8]]
NdArrayRGB8 = np.ndarray[tuple[Literal[4]], np.dtype[np.uint8]]


def _set_mario_pos_in_ram(ram: NdArrayUint8, x: int, y: int, xvel: int, yvel: int):

    # Current position.
    # E.g.:
    #   level_x=  1 pos_x=118 screen_offset= 25
    #   level_x=  4 pos_x= 45 screen_offset=106
    #   level_x=  3 pos_x=195 screen_offset=  0

    # Player horizontal position in level.  Increments for each new full screen.
    level_x = ram[0x006D]

    # Player X position.
    pos_x = ram[0x0086]

    # Player x pos within current screen offset.
    screen_offset_x = ram[0x03AD]

    # When we set "x=0", we want the player position to be at the left edge of the screen.
    # But, we may be transitioning between screen chunks.
    #
    # Example going back and forth between boundary of screen chunk:
    #   Iter: 2726   Time left: 379   level_x=  1 screen_x=  9 screen_offset=112
    #   Iter: 31238  Time left: 358   level_x=  1 screen_x= 33 screen_offset= 84
    #   Iter: 41750  Time left: 337   level_x=  0 screen_x=205 screen_offset=  0
    #   Iter: 52262  Time left: 315   level_x=  0 screen_x=205 screen_offset=  0
    #   Iter: 62774  Time left: 294   level_x=  0 screen_x=205 screen_offset=  0
    #   Iter: 73286  Time left: 273   level_x=  1 screen_x=  5 screen_offset= 56
    #   Iter: 83798  Time left: 251   level_x=  1 screen_x=  5 screen_offset= 56
    #   Iter: 94310  Time left: 230   level_x=  1 screen_x= 35 screen_offset= 80
    #   Iter: 10822  Time left: 209   level_x=  1 screen_x=182 screen_offset=111
    #   Iter: 11334  Time left: 187   level_x=  1 screen_x=249 screen_offset=112
    #   Iter: 12846  Time left: 166   level_x=  2 screen_x=  2 screen_offset=112
    #   Iter: 13358  Time left: 145   level_x=  1 screen_x=227 screen_offset= 66
    #   Iter: 14870  Time left: 123   level_x=  1 screen_x=247 screen_offset= 80
    #   Iter: 15382  Time left: 102   level_x=  2 screen_x= 11 screen_offset= 79
    #   Iter: 16894  Time left: 81    level_x=  1 screen_x=249 screen_offset= 61
    #   Iter: 17406  Time left: 59    level_x=  2 screen_x= 41 screen_offset= 83
    #
    # When the offset is greater than the screen position, we have to roll back the level.

    # Convert into an absolute left position (instead of relative to screen chunk).
    left_pos = int(level_x) * 256 + int(pos_x)

    # The left side of the screen is going to be when screen_offset_x is 0.
    new_left_pos = left_pos - int(screen_offset_x) + x

    # Recompute the position values.
    new_level_x = new_left_pos // 256
    new_pos_x = new_left_pos - (new_level_x * 256)
    new_screen_offset_x = x

    # Set the new positions.
    ram[0x006D] = new_level_x
    ram[0x0086] = new_pos_x
    ram[0x03AD] = new_screen_offset_x

    # Screen Y position.
    ram[0x00CE] = y

    # Horizontal speed.
    #   0xD8 < 0 - Moving left
    #   0x00 - Not moving
    #   0 < 0x28 - Moving right
    ram[0x0057] = 0

    # Vertical velocity, whole pixels.
    #   Upward: FB = normal jump
    #   Downward: 05 = fastest fall
    ram[0x009F] = 0

    # Vertical velocity, fractional.
    ram[0x0433] = 0


from super_mario_env import _to_controller_presses

_CONTROLLER_PRESS_TO_ARROW = {
    # Determine press:
    #   Ignore up/down.
    #   A -> up
    #   B -> longer vector
    #   right/left -> right/left
    #
    # Assume x0=0, y0=0, and we're setting x1, y1.
    tuple(_to_controller_presses([])): (0, 0),
    tuple(_to_controller_presses(['a'])): (0, 1),
    tuple(_to_controller_presses(['b'])): (0, 0),
    tuple(_to_controller_presses(['left'])): (-1, 0),
    tuple(_to_controller_presses(['right'])): (1, 0),
    tuple(_to_controller_presses(['a', 'b'])): (0, 2),
    tuple(_to_controller_presses(['a', 'left'])): (-1, 1),
    tuple(_to_controller_presses(['a', 'right'])): (1, 1),
    tuple(_to_controller_presses(['b', 'left'])): (-2, 0),
    tuple(_to_controller_presses(['b', 'right'])): (2, 0),
    tuple(_to_controller_presses(['a', 'b', 'left'])): (-2, 2),
    tuple(_to_controller_presses(['a', 'b', 'right'])): (2, 2),
}


def _get_policy_and_value_at_offset(ram: NdArrayUint8, envs: Any, device: str, agent: Any, x: int, y: int, xvel: int, yvel: int) -> tuple[float, Any]:
    # Put Mario into position and render.
    #   1. Set RAM.
    #   2. Step.
    #   3. Render.
    #   4. Put value function through.
    #   5. Restore RAM.

    env = envs.envs[0].unwrapped

    saved_state = env.nes.save()

    # Set Mario's position and velocity.
    _set_mario_pos_in_ram(ram, x=x, y=y, xvel=xvel, yvel=yvel)

    # Use no-op action.
    action_index = 0
    action_np = np.array([action_index], dtype=np.int64)

    # Step environment.
    #   next_obs.shape: torch.Size([1, 4, 84, 84])
    #   reward.shape: (1,)
    next_obs, reward, terminations, truncations, infos = envs.step(action_np)

    # Convert to pytorch Tensor
    next_obs = torch.Tensor(next_obs).to(device)

    # Retrieve value from reward.
    #   value.shape: torch.Size([1, 1])
    policy_action_probs, value = agent.get_actor_policy_probs_and_critic_value(next_obs)

    # Restore ram.
    env.nes.load(saved_state)

    value_single = value[0][0]

    return policy_action_probs, value_single, next_obs


from PIL import ImageDraw
import math

def _draw_arrow(draw: ImageDraw.Draw, start: tuple[float, float], end: tuple[float, float], head_size=1.0, shaft_length: float=3.0, color="red"):
    # Calculate direction
    dx = end[0] - start[0]
    dy = end[1] - start[1]
    angle = math.atan2(dy, dx)

    # Compute the adjusted end point for the shaft
    shaft_length = math.hypot(dx, dy) - head_size
    shaft_end = (
        start[0] + shaft_length * math.cos(angle),
        start[1] + shaft_length * math.sin(angle)
    )

    # Draw the shaft
    draw.line([start, shaft_end], fill=color, width=1)

    # Arrowhead points
    x1 = end[0] - head_size * math.cos(angle - math.pi / 6)
    y1 = end[1] - head_size * math.sin(angle - math.pi / 6)
    x2 = end[0] - head_size * math.cos(angle + math.pi / 6)
    y2 = end[1] - head_size * math.sin(angle + math.pi / 6)

    # Draw the arrowhead
    draw.polygon([end, (x1, y1), (x2, y2)], fill=color)


def _build_policy_arrows_rgb(policy_grid: list[list[tuple[float, float]]], obs_image: NdArrayRGB8):
    w = 240
    h = 224

    obs_rgba = obs_image.convert('RGBA')
    obs_rgba.putalpha(int(0.75 * 256))

    # Paste the current observation onto the image.  Take the mask from the alpha values.
    policy_rgba = Image.new('RGBA', (w, h), 'white')
    policy_rgba.paste(obs_rgba, (0, 0), mask=obs_rgba)

    # Find the scale from policy grid to pixel in image.
    scale_y = h / len(policy_grid)
    scale_x = w / len(policy_grid[0])
    scale = min(scale_x, scale_y)

    pixel_offset_x = 0.5 * scale
    pixel_offset_y = 0.5 * scale

    draw = ImageDraw.Draw(policy_rgba)

    # Image y-axis has 0 at the top, 255 at the bottom.
    flip_y = -1

    # Draw arrows on the image.
    for j, rows in enumerate(policy_grid):
        for i, arrow in enumerate(rows):
            dx, dy = arrow

            # dx, dy = 0.5, 1.0

            if False:
                r = i % 4
                if r == 0:
                    dx, dy = (1.0, 1.0)
                elif r == 1:
                    dx, dy = (2.0, 2.0)
                elif r == 2:
                    dx, dy = (-1.0, -1.0)
                elif r == 3:
                    dx, dy = (-2.0, -2.0)

            px = pixel_offset_x + i*scale
            py = pixel_offset_y + j*scale

            if arrow == (0, 0):
                pass
            else:
                _draw_arrow(
                    draw,
                    start=(px, py),
                    end=(px + dx * 0.33*scale, py + dy * 0.33*scale * flip_y),
                    head_size=0.2 * scale * 0.9,
                    shaft_length=0.8 * scale * 0.9,
                )

    return policy_rgba.convert('RGB')


def _build_values_rgb(values_grid: np.ndarray, obs_image: NdArrayRGB8) -> NdArrayRGB8:
    w = 240
    h = 224

    # Normalize values to 0-255.
    v_min = values_grid.min()
    v_max = values_grid.max()
    values_normalized = (values_grid - v_min) / (v_max - v_min)

    # Convert values to grayscale image.
    if _TEST_RANDOM_IMAGE := False:
        #values_image_np_uint8 = np.random.randint(0, 256, size=(h, w), dtype=np.uint8)
        values_image_np_uint8 = np.random.randint(200, 256, size=(h, w), dtype=np.uint8)
    else:
        values_image_np_uint8 = (values_normalized * 255).astype(np.uint8)

    values_gray = Image.fromarray(values_image_np_uint8, mode='L').resize((w, h), resample=Image.Resampling.NEAREST)

    # Convert to RGB.
    values_rgba = values_gray.convert('RGBA')
    assert values_rgba.size == (w, h), f"Unexpected values_rgba.size: {values_rgba.size} != {(w,h)}"

    # Create an image of the regular screen, with high alpha.
    if False:
        obs_np = obs[-1][-1].to(torch.uint8).numpy()
        obs_grayscale = Image.fromarray(obs_np, mode='L').resize((w, h), resample=Image.Resampling.NEAREST)
        obs_rgba = obs_grayscale.convert('RGBA')
        obs_rgba.putalpha(int(0.25 * 256))
    else:
        obs_rgba = obs_image.convert('RGBA')
        obs_rgba.putalpha(int(0.25 * 256))

    # Paste the current observation onto the image.  Take the mask from the alpha values.
    values_rgba.paste(obs_rgba, (0, 0), mask=obs_rgba)

    return values_rgba.convert('RGB')


def render_mario_pos_policy_value_sweep(envs: Any, device: str, agent: Any):
    w = 240
    h = 224
    env = envs.envs[0].unwrapped
    ram = env.nes.ram()

    # Sweep mario positions, get value function.
    x_steps = list(range(0, w, 15))
    y_steps = list(range(0, h, 15))
    num_x = len(x_steps)
    num_y = len(y_steps)
    values_grid = np.zeros((num_y, num_x), dtype=np.float64)
    policy_grid = [
        [
            (0, 0)
            for _ in range(num_x)
        ]
        for _ in range(num_y)
    ]

    # Convert actions into directions.
    #   shape = (NumActions, 2)
    action_dirs = torch.tensor([
        _CONTROLLER_PRESS_TO_ARROW[tuple(controller_press)]
        for controller_press in env.action_controller_presses
    ], device=device, dtype=torch.float32).T

    for j, y in enumerate(y_steps):
        for i, x in enumerate(x_steps):
            policy_action_probs, value, obs = _get_policy_and_value_at_offset(ram=ram, envs=envs, device=device, agent=agent, x=x, y=y, xvel=0, yvel=0)

            # Convert policy action probabilities into a direction.
            # Sum all of the direction*probability.
            policy_dir = (action_dirs * policy_action_probs).sum(axis=1)

            if False:
                print(f"SETTING POLICY DIR AT {j},{i}: {policy_dir}")

            values_grid[j][i] = value
            policy_grid[j][i] = policy_dir

    if False:
        for j, y in enumerate(y_steps):
            for i, x in enumerate(x_steps):
                print(f"POLICY[{j}][{i}]={policy_grid[j][i]}")

    obs_image = env.screen.get_image(screen_index=0)

    policy_rgb = _build_policy_arrows_rgb(policy_grid, obs_image)
    values_rgb = _build_values_rgb(values_grid, obs_image)

    return policy_rgb, values_rgb
