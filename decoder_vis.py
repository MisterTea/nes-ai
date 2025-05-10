import pygame
import numpy as np


def draw_histogram_with_axes(surface, data, rect, label, bins=50, color=(0, 200, 255), font=None):
    if font is None:
        font = pygame.font.SysFont("Arial", 14)

    x, y, w, h = rect
    hist, bin_edges = np.histogram(data, bins=bins)
    hist = hist.astype(float)
    hist /= hist.max() if hist.max() > 0 else 1

    bar_width = w / bins
    for i in range(bins):
        bar_height = hist[i] * h
        bar_x = x + i * bar_width
        bar_y = y + h - bar_height
        pygame.draw.rect(surface, color, pygame.Rect(bar_x, bar_y, bar_width, bar_height))

    # Axes
    axis_color = (200, 200, 200)
    pygame.draw.line(surface, axis_color, (x, y + h), (x + w, y + h), 1)
    pygame.draw.line(surface, axis_color, (x, y), (x, y + h), 1)

    # Label
    label_surf = font.render(label, True, axis_color)
    surface.blit(label_surf, (x + 5, y - 18))

    # Tick values
    min_val, max_val = bin_edges[0], bin_edges[-1]
    min_tick = font.render(f"{min_val:.2f}", True, axis_color)
    max_tick = font.render(f"{max_val:.2f}", True, axis_color)
    surface.blit(min_tick, (x, y + h + 2))
    surface.blit(max_tick, (x + w - max_tick.get_width(), y + h + 2))
