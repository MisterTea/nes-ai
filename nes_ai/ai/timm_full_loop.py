import logging
import os
import time
from pathlib import Path

import click
import numpy

from nes_ai.ai.timm_rl import train_rl

print(numpy.get_include())


def get_most_recent_file(directory):
    """Gets the most recently created file in a directory."""

    files = [
        f for f in os.listdir(directory) if os.path.isfile(os.path.join(directory, f))
    ]
    if not files:
        return None

    newest_file = max(files, key=lambda f: os.path.getctime(os.path.join(directory, f)))
    return os.path.join(directory, newest_file)


def run_sim():
    import Cython.Compiler.Options
    import pygame
    from Cython.Build import cythonize
    from setuptools import Extension, find_packages, setup

    Cython.Compiler.Options.annotate = True

    extensions = [
        Extension("cycore.*", ["nes/cycore/*.pyx"], include_dirs=[numpy.get_include()])
    ]
    extensions = cythonize(
        extensions,
        compiler_directives={
            "language_level": 3,
            "profile": False,
            "boundscheck": False,
            "nonecheck": False,
            "cdivision": True,
        },
        annotate=True,
    )

    import pyximport

    pyximport.install(
        setup_args={"include_dirs": numpy.get_include()}, reload_support=True
    )

    from nes import NES, SYNC_AUDIO, SYNC_NONE, SYNC_PYGAME, SYNC_VSYNC
    from nes.ai_handler import AiHandler, LearnMode

    epoch = 0

    model = get_most_recent_file("timm_rl_models/")
    print(model)

    data_path = Path(f"data/1_1_rl_{epoch}")

    nes = NES(
        "./roms/Super_mario_brothers.nes",
        AiHandler(
            data_path,
            LearnMode.RL,
            score_model=model,
            bootstrap_expert_path=Path("data/1_1_expert"),
        ),
        sync_mode=SYNC_PYGAME,
        opengl=True,
        audio=False,
    )

    nes.run()
    pygame.display.quit()
    pygame.quit()


@click.command()
def main():
    epoch = 0

    run_sim()

    data_path = Path(f"data/1_1_rl_{epoch}")
    model = get_most_recent_file("timm_rl_models/")
    train_rl(data_path, model)


if __name__ == "__main__":
    main()
