import numpy
from Cython.Build import cythonize
from setuptools import Extension, find_packages, setup

extensions = [
    Extension("mcts", ["nes_ai/mcts/*.pyx"], include_dirs=[numpy.get_include()])
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

pyximport.install(setup_args={"include_dirs": numpy.get_include()}, reload_support=True)

print("DONE")

from nes_ai.mcts.mcts import MCTS
from nes_ai.mcts.tictactoe import TicTacToeState

if __name__ == "__main__":
    mcts = MCTS(exploration_bias=0.43)  # This bias was chosen empirically.

    state = TicTacToeState()
    print(state)

    while not state.is_terminal():
        if state.get_current_team() == 1:
            # Player's turn: Prompt for a move.
            moves = state.get_legal_moves()
            for i, move in enumerate(moves):
                print(f"{i+1}. {str(move)}")
            move = moves[int(input("> ")) - 1]
            state = state.make_move(move)
        else:
            # AI's turn: Use MCTS to find the best move.
            state = mcts.search(state, max_iterations=10000)

        print(state)
