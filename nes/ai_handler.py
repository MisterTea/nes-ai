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


class AiHandler:
    def __init__(
        self,
        data_path: Path,
        learn_mode: LearnMode,
        bootstrap_expert_path: Path | None = None,
        score_model: Path | None = None,
    ):
        self.rollout_data = RolloutData(data_path)

        self.controller_buffer = torch.zeros((3, 8), dtype=torch.float)
        self.screen_buffer = torch.zeros((4, 3, 224, 224), dtype=torch.float)
        self.last_reward_map = None
        self.reward_map = None
        self.score_model = score_model
        print("Scoring with model", score_model)

        self.expert_dataset = None
        self.learn_mode = learn_mode
        if learn_mode == LearnMode.IMITATION_VALIDATION:
            self.expert_dataset = NESDataset(
                bootstrap_expert_path, train=False, imitation_learning=True
            )
        if bootstrap_expert_path is not None:
            self.bootstrap_expert_data = RolloutData(
                bootstrap_expert_path, readonly=True
            )

        self.frames_until_exit = -1

    def shutdown(self):
        print("Shutting down ai handler")
        self.rollout_data.close()

    def update(self, frame, controller1, ram, screen_buffer_image):
        self.last_reward_map = self.reward_map
        self.reward_map, reward_vector = compute_reward_map(
            self.last_reward_map, torch.from_numpy(ram).int()
        )
        if self.frames_until_exit > 0:
            self.frames_until_exit -= 1
            if self.frames_until_exit == 0:
                print("REACHED END")
                return False
        else:
            if frame > 200:
                if self.reward_map.lives < 2:
                    print("LOST LIFE")
                    self.frames_until_exit = 10
                if self.reward_map.level > 0:
                    print("GOT NEW LEVEL")
                    self.frames_until_exit = 300

        print(
            "TIME LEFT", self.reward_map.time_left, "LEFT POS", self.reward_map.left_pos
        )
        minimum_progress = (400 - self.reward_map.time_left) * 2
        if frame > 200 and self.reward_map.left_pos < minimum_progress:
            print("TOO LITTLE PROGRESS, DYING")
            ram[0x75A] = 1
        # print("FALLING IN PIT", torch.from_numpy(ram).int()[0x0712])
        # print("REWARD", self.reward_map, reward_vector)

        # print("CONTROLLER", controller1.is_pressed)

        with torch.no_grad():
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
            # print("Environment Image ", frame, image.mean())
            for buffer_i in range(3):
                self.screen_buffer[buffer_i, :, :, :] = self.screen_buffer[
                    buffer_i + 1, :, :, :
                ]
            self.screen_buffer[3, :, :, :] = image[:, :, :]

            if self.learn_mode == LearnMode.REPLAY_IMITATION:
                # Replay
                controller = self.rollout_data.expert_controller[str(frame)]
                controller1.set_state(controller)
                assert (
                    self.rollout_data.reward_map_history[str(frame)] == self.reward_map
                ), f"{self.rollout_data.reward_map_history[str(frame)]} != {self.reward_map}"
                print("REWARD VECTOR", reward_vector)
                self.rollout_data.reward_vector_history[str(frame)] = reward_vector

            if self.learn_mode == LearnMode.IMITATION_VALIDATION:
                # Replay
                controller = self.bootstrap_expert_data.expert_controller[str(frame)]
                assert isinstance(controller, list), f"{type(controller)}"
                controller1.set_state(controller)
                assert (
                    self.bootstrap_expert_data.reward_map_history[str(frame)]
                    == self.reward_map
                ), f"{self.bootstrap_expert_data.reward_map_history[str(frame)]} != {self.reward_map}"
                print("REWARD VECTOR", reward_vector)
                self.rollout_data.reward_vector_history[str(frame)] = reward_vector

                if frame >= 200:
                    # Score model timm

                    print(self.screen_buffer.shape, image.shape)
                    reward_history = self._get_reward_history(frame)

                    action, log_prob, entropy, value = score(
                        self.score_model,
                        self.screen_buffer,
                        self.controller_buffer,
                        reward_history,
                        str(frame),
                    )

                    if frame >= 205:  # Check against ground truth
                        (
                            image_stack,
                            value,
                            past_inputs,
                            past_rewards,
                            label_int,
                            recorded_action_log_prob,
                            advantages,
                        ) = self.expert_dataset.get_frame(frame)
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

                        logged_action, logged_log_prob, logged_entropy, logged_value = (
                            score(
                                self.score_model,
                                image_stack,
                                past_inputs,
                                past_rewards,
                                str(frame),
                            )
                        )
                        if not torch.equal(action, logged_action):
                            print(f"Actions don't match: {action} != {logged_action}")
                        assert torch.equal(
                            log_prob, logged_log_prob
                        ), f"{log_prob} != {logged_log_prob}"
                        assert torch.equal(
                            entropy, logged_entropy
                        ), f"{entropy} != {logged_entropy}"

                    self.rollout_data.agent_params[str(frame)] = {
                        "action": action.tolist(),
                        "log_prob": float(log_prob.item()),
                        "entropy": float(entropy.item()),
                        "value": value.tolist(),
                    }
                    controller1.update()
                    if not controller1.is_any_pressed():
                        print(action)
                        controller1.set_state(action)

                    # Overwrite with expert
                    expert_controller_for_frame = (
                        self.bootstrap_expert_data.expert_controller_no_start_select(
                            str(frame)
                        )
                    )
                    if not torch.equal(
                        torch.IntTensor(expert_controller_for_frame).to(
                            device=action.device
                        ),
                        action,
                    ):
                        print("WRONG ANSWER")
                        controller1.set_state(expert_controller_for_frame)

                else:
                    # Use expert for intro
                    controller = self.bootstrap_expert_data.expert_controller[
                        str(frame)
                    ]
                    controller1.set_state(controller)

            if self.learn_mode == LearnMode.RL:
                if frame == 0:
                    self.rollout_data.clear()

                if frame >= 200:
                    # Score model timm
                    reward_history = self._get_reward_history(frame)

                    action, log_prob, entropy, value = score(
                        self.score_model,
                        self.screen_buffer,
                        self.controller_buffer,
                        reward_history,
                        str(frame),
                    )
                    print("LOGPROB", log_prob)
                    self.rollout_data.agent_params[str(frame)] = {
                        "action": action.cpu().tolist(),
                        "log_prob": float(log_prob.item()),
                        "entropy": float(entropy.item()),
                        "value": value.cpu().tolist(),
                        "screen_buffer": self.screen_buffer.cpu().tolist(),
                        "controller_buffer": self.controller_buffer.cpu().tolist(),
                        "reward_history": reward_history.cpu().tolist(),
                    }
                    controller1.update()
                    if not controller1.is_any_pressed():
                        # Human isn't taking over
                        controller1.set_state(action.tolist())

                else:
                    # Use expert for intro
                    controller = self.bootstrap_expert_data.expert_controller[
                        str(frame)
                    ]
                    controller1.set_state(controller)

            if (
                self.learn_mode == LearnMode.DATA_COLLECT
                or self.learn_mode == LearnMode.IMITATION_VALIDATION
                or self.learn_mode == LearnMode.RL
            ):
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

            for buffer_i in range(2):
                self.controller_buffer[buffer_i, :] = self.controller_buffer[
                    buffer_i + 1, :
                ]
            self.controller_buffer[2, :] = torch.FloatTensor(controller1.get_ai_state())

        return True

    def _get_reward_history(self, frame):
        reward_history = torch.zeros((3, REWARD_VECTOR_SIZE), dtype=torch.float)
        for x in range(3):
            reward_history[x, :] = torch.FloatTensor(
                self.rollout_data.reward_vector_history[str((frame - 3) + x)]
            )
        return reward_history
