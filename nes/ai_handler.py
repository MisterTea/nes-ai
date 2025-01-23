import shelve
from enum import Enum
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from nes_ai.ai.base import SELECT, START, compute_reward_map
from nes_ai.ai.nes_dataset import NESDataset
from nes_ai.ai.rollout_data import RolloutData
from nes_ai.ai.score_model import score
from nes_ai.ai.timm_imitation_learning import REWARD_VECTOR_SIZE


class LearnMode(Enum):
    DATA_COLLECT = 1
    REPLAY_IMITATION = 2
    IMITATION_LEARNING = 3
    IMITATION_VALIDATION = 4
    RL = 5


current_learn_mode = LearnMode.REPLAY_IMITATION


class AiHandler:
    def __init__(self, data_path: Path):
        self.rollout_data = RolloutData(data_path)

        self.controller_buffer = torch.zeros((3, 8), dtype=torch.float)
        self.screen_buffer = torch.zeros((4, 3, 224, 224), dtype=torch.float)
        self.last_reward_map = None
        self.reward_map = None

        self.expert_dataset = None
        if current_learn_mode == LearnMode.IMITATION_VALIDATION:
            self.expert_dataset = NESDataset(
                data_path, train=False, imitation_learning=True
            )

    def shutdown(self):
        self.rollout_data.close()

    def update(self, frame, controller1, ram, screen_buffer_image):
        self.last_reward_map = self.reward_map
        self.reward_map, reward_vector = compute_reward_map(
            self.last_reward_map, torch.from_numpy(ram).int()
        )
        print("REWARD", self.reward_map, reward_vector)

        # print("CONTROLLER", controller1.is_pressed)

        if True:
            from torchvision import transforms

            DEFAULT_TRANSFORM = transforms.Compose(
                [
                    transforms.ToTensor(),
                    transforms.Resize((224, 224)),
                    transforms.Normalize(
                        mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                    ),
                ]
            )
            image = DEFAULT_TRANSFORM(screen_buffer_image)
            print("Environment Image ", frame, image.mean())
            for buffer_i in range(3):
                self.screen_buffer[buffer_i, :, :, :] = self.screen_buffer[
                    buffer_i + 1, :, :, :
                ]
            self.screen_buffer[3, :, :, :] = image[:, :, :]

            if current_learn_mode == LearnMode.DATA_COLLECT:
                if frame == 0:
                    print("RESETTING")
                    self.rollout_data.input_images.clear()
                    self.rollout_data.expert_controller.clear()
                    self.rollout_data.reward_map_history.clear()
                    self.rollout_data.reward_vector_history.clear()
                    self.rollout_data.agent_params.clear()

                # Data Collect
                controller_pressed = controller1.get_ai_state()
                self.rollout_data.expert_controller[str(frame)] = controller_pressed
                self.rollout_data.reward_map_history[str(frame)] = self.reward_map
                self.rollout_data.reward_vector_history[str(frame)] = reward_vector
                self.rollout_data.put_image(screen_buffer_image, frame)
                if frame % 60 == 0:
                    print("SAVING")
                    self.rollout_data.sync()

            if current_learn_mode == LearnMode.REPLAY_IMITATION:
                # Replay
                controller = self.rollout_data.expert_controller[str(frame)]
                controller1.set_state(controller)
                assert (
                    self.rollout_data.reward_map_history[str(frame)] == self.reward_map
                ), f"{self.rollout_data.reward_map_history[str(frame)]} != {self.reward_map}"
                print("REWARD VECTOR", reward_vector)
                self.rollout_data.reward_vector_history[str(frame)] = reward_vector

            if current_learn_mode == LearnMode.IMITATION_VALIDATION:
                # Replay
                controller = self.rollout_data.expert_controller[str(frame)]
                controller1.set_state(controller)
                assert (
                    self.rollout_data.reward_map_history[str(frame)] == self.reward_map
                ), f"{self.rollout_data.reward_map_history[str(frame)]} != {self.reward_map}"
                print("REWARD VECTOR", reward_vector)
                self.rollout_data.reward_vector_history[str(frame)] = reward_vector

                if frame > 200:
                    # Score model timm

                    print(self.screen_buffer.shape, image.shape)
                    reward_history = self._get_reward_history(frame)

                    if frame >= 205:  # Check against ground truth
                        image_stack, value, past_inputs, past_rewards, label_int = (
                            self.expert_dataset.get_frame(frame)
                        )
                        assert torch.equal(
                            past_inputs, self.controller_buffer
                        ), f"{past_inputs} != {self.controller_buffer}"
                        if not torch.equal(image_stack, self.screen_buffer):
                            print(image_stack[3].mean(), self.screen_buffer[3].mean())
                            assert torch.equal(
                                image_stack[0], self.screen_buffer[0]
                            ), f"{image_stack[0]} != {self.screen_buffer[0]}"
                            assert torch.equal(
                                image_stack[1], self.screen_buffer[1]
                            ), f"{image_stack[1]} != {self.screen_buffer[1]}"
                            assert torch.equal(
                                image_stack[2], self.screen_buffer[2]
                            ), f"{image_stack[2]} != {self.screen_buffer[2]}"
                            assert torch.equal(
                                image_stack[3], self.screen_buffer[3]
                            ), f"{image_stack[3]} != {self.screen_buffer[3]}"

                    action, log_prob, entropy, value = score(
                        self.screen_buffer,
                        self.controller_buffer,
                        reward_history,
                        str(frame),
                    )
                    self.rollout_data.agent_params[str(frame)] = {
                        "action": action.numpy(),
                        "log_prob": float(log_prob.item()),
                        "entropy": float(entropy.item()),
                        "value": value.numpy(),
                    }
                    controller1.update()
                    if not controller1.is_any_pressed():
                        print(action)
                        controller1.set_state(action)

                    # Overwrite with expert
                    expert_controller_for_frame = (
                        self.rollout_data.expert_controller_no_start_select(str(frame))
                    )
                    if not torch.equal(
                        torch.IntTensor(expert_controller_for_frame), action
                    ):
                        print("WRONG ANSWER")
                        controller1.set_state(expert_controller_for_frame)

                else:
                    # Use expert for intro
                    controller = self.rollout_data.expert_controller[str(frame)]
                    controller1.set_state(controller)

            if current_learn_mode == LearnMode.RL:
                if frame == 0:
                    self.rollout_data.agent_params.clear()

                if frame > 200:
                    # Score model timm
                    print(self.screen_buffer.shape, image.shape)
                    reward_history = self._get_reward_history(frame)

                    action, log_prob, entropy, value = score(
                        self.screen_buffer,
                        self.controller_buffer,
                        reward_history,
                        str(frame),
                    )
                    self.rollout_data.agent_params[str(frame)] = {
                        "action": action.numpy(),
                        "log_prob": float(log_prob.item()),
                        "entropy": float(entropy.item()),
                        "value": value.numpy(),
                    }
                    controller1.update()
                    if not controller1.is_any_pressed():
                        # Allow human to take over
                        print(action)
                        controller1.set_state(action)

                else:
                    # Use expert for intro
                    controller = self.rollout_data.expert_controller[str(frame)]
                    controller1.set_state(controller)

            for buffer_i in range(2):
                self.controller_buffer[buffer_i, :] = self.controller_buffer[
                    buffer_i + 1, :
                ]
            self.controller_buffer[2, :] = torch.FloatTensor(controller1.get_ai_state())

    def _get_reward_history(self, frame):
        reward_history = torch.zeros((3, REWARD_VECTOR_SIZE), dtype=torch.float)
        for x in range(3):
            reward_history[x, :] = torch.FloatTensor(
                self.rollout_data.reward_vector_history[str((frame - 3) + x)]
            )
        return reward_history
