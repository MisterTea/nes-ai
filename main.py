from pathlib import Path

import Cython.Compiler.Options
from Cython.Build import cythonize
from setuptools import Extension, find_packages, setup

Cython.Compiler.Options.annotate = True
import numpy

from nes.ai_handler import AiHandler, LearnMode

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


import numpy

print(numpy.get_include())

import pyximport

pyximport.install(setup_args={"include_dirs": numpy.get_include()}, reload_support=True)

import logging

import click

from nes import NES, SYNC_AUDIO, SYNC_NONE, SYNC_PYGAME, SYNC_VSYNC


@click.command()
@click.option(
    "--learn-mode",
    required=True,
    type=click.Choice(["RL", "DATA_COLLECT", "IMITATION_VALIDATION"]),
)
@click.option("--score-model", required=False)
def main(learn_mode: LearnMode, score_model: Path | None = None):

    if learn_mode == "RL":
        nes = NES(
            "./roms/Super_mario_brothers.nes",
            AiHandler(
                Path("data/1_1_rl_2"),
                LearnMode.RL,
                score_model=score_model,
                bootstrap_expert_path=Path("data/1_1_expert"),
            ),
            sync_mode=SYNC_PYGAME,
            opengl=True,
            audio=False,
        )

    if learn_mode == "IMITATION_VALIDATION":
        nes = NES(
            "./roms/Super_mario_brothers.nes",
            AiHandler(
                Path("data/1_1_rl"),
                LearnMode.IMITATION_VALIDATION,
                bootstrap_expert_path=Path("data/1_1_expert"),
                score_model=score_model,
            ),
            sync_mode=SYNC_PYGAME,
            opengl=True,
            audio=False,
        )

    if learn_mode == "DATA_COLLECT":
        nes = NES(
            "./roms/Super_mario_brothers.nes",
            AiHandler(Path("data/1_1_expert"), LearnMode.DATA_COLLECT),
            sync_mode=SYNC_PYGAME,
            opengl=True,
            audio=False,
        )

    nes.run()


if __name__ == "__main__":
    main()
