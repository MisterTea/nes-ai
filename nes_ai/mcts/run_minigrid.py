import logging
import pickle

import numpy
import pyximport

from nes_ai import profiler

pyximport.install(setup_args={"include_dirs": numpy.get_include()}, reload_support=True)

import copy

from minigrid.envs import EmptyEnv

from nes_ai.mcts.mcts import MCTS, GameState


class DumpableEnv(EmptyEnv):
    def dumps(self):
        return pickle.dumps(
            # Current position and direction of the agent
            (
                self.agent_pos,
                self.agent_dir,
                # Current grid and mission and carrying
                self.grid,
                self.carrying,
                self.step_count,
                self.carrying,
            )
        )

    def loads(self, data):
        (
            self.agent_pos,
            self.agent_dir,
            # Current grid and mission and carrying
            self.grid,
            self.carrying,
            self.step_count,
            self.carrying,
        ) = pickle.loads(data)


env = DumpableEnv()


class GymEnvState(GameState):
    def __init__(self, terminated=False, reward=0):
        self.terminated = terminated
        self.reward = reward
        self.state = env.dumps()

    def get_current_team(self):
        return 1

    def get_legal_moves(self):
        return list(range(3))

    def refresh_env(self):
        env.loads(self.state)

    def make_move(self, move):
        assert (
            env.dumps() == self.state
        ), f"State mismatch: {env.dumps()} != {self.state}"
        env.loads(self.state)
        obs, reward, terminated, truncated, _ = env.step(move)
        # if terminated:
        #     print(f"Episode terminated with reward: {self.reward + reward}")
        # if truncated:
        #     print(f"Episode truncated with reward: {self.reward + reward}")
        new_state = GymEnvState(terminated or truncated, self.reward + reward)
        return new_state

    def is_terminal(self):
        return self.terminated

    def get_reward(self):
        return self.reward

    def __repr__(self):
        env.loads(self.state)
        return env.pprint_grid()


def main():
    mcts = MCTS(exploration_bias=0.43)  # This bias was chosen empirically.

    global env
    env.reset()

    state = GymEnvState()
    print(state)

    while not state.is_terminal():
        with profiler.profile(print_session=True, profile_memory=False):
            move = mcts.search(state, max_iterations=1000, return_type="move")
        print(f"Selected move: {move}")
        state.refresh_env()
        state = state.make_move(move)
        print(state)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()
