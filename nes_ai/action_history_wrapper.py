from collections import deque

import gymnasium as gym
import numpy as np
from gymnasium import spaces
from gymnasium.core import ActType, ObsType, WrapperObsType
from gymnasium.wrappers import TransformObservation


class ActionHistoryWrapper(
    TransformObservation[WrapperObsType, ActType, ObsType],
    gym.utils.RecordConstructorArgs,
):

    def __init__(self, env: gym.Env[ObsType, ActType], history_length: int):
        assert isinstance(env.observation_space, spaces.Box)

        self.history_length = history_length
        self.action_history = deque(maxlen=history_length)
        for x in range(history_length):
            self.action_history.append(np.zeros(2, dtype=np.uint8))

        assert len(self.action_history) == history_length
        assert len(self.action_history[0]) == 2

        new_observation_space = spaces.Tuple(
            (
                spaces.Box(
                    low=0,
                    high=255,
                    shape=env.observation_space.shape,
                    dtype=np.uint8,
                ),
                spaces.Sequence(spaces.Discrete(2)),
            ),
        )

        gym.utils.RecordConstructorArgs.__init__(self, history_length=history_length)
        TransformObservation.__init__(
            self,
            env=env,
            func=(lambda obs: (obs, np.array(list(self.action_history)).reshape(6))),
            observation_space=new_observation_space,
        )

    def step(self, action):
        # Store the current action in the action history
        self.action_history.append(action)

        # Call the step method of the base environment
        return super().step(action)

    def reset(self, **kwargs):
        # Reset the action history when the environment resets
        self.action_history.clear()
        for x in range(self.history_length):
            self.action_history.append(np.zeros(2, dtype=np.uint8))

        return super().reset(**kwargs)
