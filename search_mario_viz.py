from typing import Any, Literal

import numpy as np
from PIL import Image, ImageDraw

from super_mario_env_search import SCREEN_W, SCREEN_H

NdArrayUint8 = np.ndarray[np.dtype[np.uint8]]
NdArrayRGB8 = np.ndarray[tuple[Literal[3]], np.dtype[np.uint8]]
PatchId = Any


_SPACE_R = 0


def optimal_patch_layout(screen_width: int, screen_height: int, n_patches: int) -> tuple[int, int, int]:
    max_patch_size = 0
    best_cols = best_rows = None

    # Try all possible number of columns from 1 to n_patches (but can't exceed screen_width)
    for n_cols in range(1, n_patches + 1):
        patch_size = screen_width // n_cols
        n_rows = int(np.ceil(n_patches / n_cols))
        total_height = patch_size * n_rows

        if total_height <= screen_height:
            # Maximize patch size
            if patch_size > max_patch_size:
                max_patch_size = patch_size
                best_cols = n_cols
                best_rows = n_rows

    return (best_rows, best_cols, max_patch_size)


def build_patch_histogram_rgb(
    patch_id_and_weight_pairs: list[PatchId, int],
    current_patch: PatchId,
    patch_size: int,
    max_patch_x: int,
    max_patch_y: int,
    hist_rows: int,
    hist_cols: int,
    pixel_size: int,
) -> NdArrayRGB8:
    hr, hc = hist_rows, hist_cols

    patch_histogram = np.zeros((hr + 1, hc + 1), dtype=np.float64)

    if False:
        patch_id_and_weight_pairs = list(patch_id_and_weight_pairs)
        print(f"FIRST patch_id_and_weight_pairs: {patch_id_and_weight_pairs[0]}")
        raise AssertionError("DEBUG")

    special_section_offsets = {}
    next_special_section_id = [0]

    # Sometimes we get a discontinuous jump, like:
    #   Discountinuous x position: 1013 -> 65526
    #
    # Seems to happen when crossing boundaries in discontinuous levels like 4-4.
    def _calc_c_for_special_section(check_patch_x: PatchId, hr: int, hc: int):
        # print(f"REWRITING SPECIAL SECTION: patch={check_patch_x.patch_x},{check_patch_x.patch_y} hr={hr} hc={hc}")
        # TODO(millman): this isn't right, special section is overlapping other things

        # The section here is the 255-width pixel section that the level is divided into.
        # Also called "screen" in the memory map.
        x = check_patch_x.patch_x * patch_size

        section_x = (x // 256) * 256
        section_patch_x = section_x // patch_size

        offset_x = x - section_x
        offset_patch_x = offset_x // patch_size

        # Determine how many full-screen offsets we need from the edge of the histogram.
        if section_patch_x not in special_section_offsets:
            special_section_offsets[section_patch_x] = next_special_section_id[0]
            next_special_section_id[0] += 1

        section_id = special_section_offsets[section_patch_x]
        patches_in_section = 256 // patch_size

        # Calculate the starting patch of the section.
        section_c = hc - (1 + section_id) * patches_in_section

        # Calculate how much offset we need from the start of the section.
        # Rewrite the x and y position, for display into this after-level section.
        c = section_c + offset_patch_x

        # print(f"REWRITING SPECIAL SECTION: patch={check_patch_x.patch_x},{check_patch_x.patch_y} hr={hr} hc={hc} -> c={c}")

        return c


    for save_patch_id, weight in patch_id_and_weight_pairs:
        patch_x, patch_y = save_patch_id.patch_x, save_patch_id.patch_y

        # For display purposes only, if we're at one of the special offscreen locations,
        # Use an offset, but still show it in the histogram.
        if patch_x <= max_patch_x:
            # What row of the level we're in.  Wrap around if past the end of the screen.
            wrap_i = patch_x // hc

            r = wrap_i * (max_patch_y + _SPACE_R) + patch_y
            c = patch_x % hc
        else:
            # Special case, we're past the end of the level for some special section.
            r = hr - max_patch_y + patch_y
            c = _calc_c_for_special_section(save_patch_id, hr=hr, hc=hc)

        try:
            patch_histogram[r][c] = weight
        except IndexError:
            print(f"PATCH LAYOUT: max_patches_x={max_patch_x} max_patches_y={max_patch_y} pixel_size={pixel_size} hr={hr} hc={hc}")
            print(f"BAD CALC? wrap_i={wrap_i} hr={hr} hc={hc} r={r} c={c} patch_x={patch_x} patch_y={patch_y}")

            patch_histogram[hr][hc] += 1

    #print(f"HISTOGRAM min={patch_histogram.min()} max={patch_histogram.max()}")

    w_zero = patch_histogram == 0

    # Normalize counts to range (0, 255)
    grid_f = patch_histogram - patch_histogram.min()
    grid_g = (grid_f / grid_f.max() * 255).astype(np.uint8)

    # Reset untouched patches to zero.
    grid_g[w_zero] = 0

    # Convert to RGB,
    grid_rgb = np.stack([grid_g]*3, axis=-1)

    # Mark current patch.
    if current_patch.patch_x <= max_patch_x:
        px, py = current_patch.patch_x, current_patch.patch_y
        wrap_i = px // hc
        patch_r = wrap_i * (max_patch_y + _SPACE_R) + py
        patch_c = px % hc
    else:
        # Special case, we're past the end of the level for some special section.
        patch_r = hr - max_patch_y + current_patch.patch_y
        patch_c = _calc_c_for_special_section(current_patch, hr=hr, hc=hc)

    try:
        grid_rgb[patch_r][patch_c] = (0, 255, 0)
    except IndexError:
        print(f"PATCH LAYOUT: max_patches_x={max_patch_x} max_patches_y={max_patch_y} pixel_size={pixel_size} hr={hr} hc={hc}")
        print(f"BAD CALC? wrap_i={wrap_i} hr={hr} hc={hc} r={r} c={c} patch_x={patch_x} patch_y={patch_y}")

        grid_rgb[hr][hc] = (0, 255, 0)

    # Convert to screen.
    img_gray = Image.fromarray(grid_rgb, mode='RGB')
    img_rgb_240 = img_gray.resize((SCREEN_W, SCREEN_H), resample=Image.NEAREST)

    return img_rgb_240


def draw_patch_path(
    img_rgb_240: NdArrayRGB8,
    patch_history: list[PatchId],
    patch_size: int,
    max_patch_x: int,
    max_patch_y: int,
    hist_rows: int,
    hist_cols: int,
    pixel_size: int,
) -> NdArrayRGB8:
    hr, hc = hist_rows, hist_cols

    special_section_offsets = {}
    next_special_section_id = [0]

    # Sometimes we get a discontinuous jump, like:
    #   Discountinuous x position: 1013 -> 65526
    #
    # Seems to happen when crossing boundaries in discontinuous levels like 4-4.
    def _calc_c_for_special_section(check_patch_x: PatchId, hr: int, hc: int):
        # print(f"REWRITING SPECIAL SECTION: patch={check_patch_x.patch_x},{check_patch_x.patch_y} hr={hr} hc={hc}")
        # TODO(millman): this isn't right, special section is overlapping other things

        # The section here is the 255-width pixel section that the level is divided into.
        # Also called "screen" in the memory map.
        x = check_patch_x.patch_x * patch_size

        section_x = (x // 256) * 256
        section_patch_x = section_x // patch_size

        offset_x = x - section_x
        offset_patch_x = offset_x // patch_size

        # Determine how many full-screen offsets we need from the edge of the histogram.
        if section_patch_x not in special_section_offsets:
            special_section_offsets[section_patch_x] = next_special_section_id[0]
            next_special_section_id[0] += 1

        section_id = special_section_offsets[section_patch_x]
        patches_in_section = 256 // patch_size

        # Calculate the starting patch of the section.
        section_c = hc - (1 + section_id) * patches_in_section

        # Calculate how much offset we need from the start of the section.
        # Rewrite the x and y position, for display into this after-level section.
        c = section_c + offset_patch_x

        # print(f"REWRITING SPECIAL SECTION: patch={check_patch_x.patch_x},{check_patch_x.patch_y} hr={hr} hc={hc} -> c={c}")

        return c

    draw = ImageDraw.Draw(img_rgb_240)

    # print(f"PATCH HISTORY TO DRAW: {patch_history}")
    prev_xy = None
    prev_special = False
    for i, p in enumerate(patch_history):
        patch_x, patch_y = p.patch_x, p.patch_y

        # For display purposes only, if we're at one of the special offscreen locations,
        # Use an offset, but still show it in the histogram.
        if patch_x <= max_patch_x:
            # What row of the level we're in.  Wrap around if past the end of the screen.
            wrap_i = patch_x // hc

            r = wrap_i * (max_patch_y + _SPACE_R) + patch_y
            c = patch_x % hc

            is_special = False
        else:
            # Special case, we're past the end of the level for some special section.
            r = hr - max_patch_y + patch_y
            c = _calc_c_for_special_section(p, hr=hr, hc=hc)

            is_special = True

        # Convert r,c patch to center of pixel for patch.
        #  pixel size: 4
        #  hr = 60
        #  screen_h = 224
        #
        #  hr * pixel_size = 224
        #  hc * pixel_size = 240
        scale_x = SCREEN_W / ((hc + 1) * pixel_size)
        scale_y = SCREEN_H / ((hr + 1) * pixel_size)
        y = (r + 0.5) * pixel_size * scale_y
        x = (c + 0.5) * pixel_size * scale_x

        if prev_xy is None or prev_special != is_special:
            draw.line([(x,y), (x,y)], fill=(0, 255, 0), width=1)
        else:
            draw.line([prev_xy, (x,y)], fill=(0, 255, 0), width=1)

        prev_xy = x, y
        prev_special = is_special

    return img_rgb_240


def draw_patch_grid(
    img_rgb_240: NdArrayRGB8,
    patch_size: int,
    ram: NdArrayUint8,
    x: int,
    y: int,
) -> NdArrayRGB8:

    # Player x pos within current screen offset.
    screen_offset_x = ram[0x03AD]

    # Patches are based on player position, but the view is based on screen position.
    # The first patch is going to be offset from the left side of the screen.
    left_x = x - int(screen_offset_x)

    # The first vertical line is going to be at an offset:
    #   if the viewport is at 0, then the offset is 0
    #   if the viewport is at 10, then the first patch will be drawn at 32-10 or 22
    #   if the viewport is at 100, then first patch will be drawn at 32 - (100 % 32) or 28

    x_offset = patch_size - left_x % patch_size
    y_offset = 0

    draw = ImageDraw.Draw(img_rgb_240)

    # Draw all horizontal lines.
    for j in range(SCREEN_H // patch_size + 1):
        x0 = 0
        x1 = SCREEN_W
        y0 = j * patch_size + y_offset
        draw.line([(x0, y0), (x1, y0)], fill=(0, 255, 0), width=1)

    # Draw all vertical lines.
    for i in range(SCREEN_W // patch_size + 1):
        y0 = 0
        y1 = SCREEN_H
        x0 = x_offset + i * patch_size
        draw.line([(x0, y0), (x0, y1)], fill=(0, 255, 0), width=1)

    return img_rgb_240
