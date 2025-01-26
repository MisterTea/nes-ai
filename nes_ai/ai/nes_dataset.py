import io
import json

import torch
from PIL import Image
from torchvision import transforms

from nes_ai.ai.helpers import upscale_and_get_labels
from nes_ai.ai.rollout_data import RolloutData

DEFAULT_TRANSFORM = transforms.Compose(
    [
        transforms.ToTensor(),
        transforms.Resize((224, 224)),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]
)

GAMMA = 0.995
GAE_LAMBDA = 0.95

FRAMES_TO_SKIP = 200


class NESDataset(torch.utils.data.Dataset):
    def __init__(self, data_path, train: bool, imitation_learning: bool):
        self.train = train

        self.bootstrapped = False
        self.imitation_learning = imitation_learning
        self.data_path = data_path

    def __len__(self):
        rollout_data = RolloutData(self.data_path, readonly=True)
        end_frame = NESDataset._compute_end_frame(rollout_data)
        rollout_data.close()
        return end_frame - FRAMES_TO_SKIP

    def get_image_and_input_at_frame(self, frame):
        # print(
        #     "image frames", self.data_path, list(self.rollout_data.input_images.keys())
        # )
        image = self.rollout_data.get_image(str(frame))
        image = DEFAULT_TRANSFORM(image)
        # print("Replay buffer image",frame, image.mean())
        # print("Getting image and input at frame", frame, image_filename, self.file_label_map[image_filename.name])
        return image, self.rollout_data.expert_controller_no_start_select(str(frame))

    @staticmethod
    def _compute_end_frame(rollout_data):
        end_frame = FRAMES_TO_SKIP
        while str(end_frame) in rollout_data.input_images:
            end_frame += 1
        assert end_frame > FRAMES_TO_SKIP, f"{list(rollout_data.input_images.keys())}"
        return end_frame

    def bootstrap(self):
        self.rollout_data = RolloutData(self.data_path)

        self.end_frame = NESDataset._compute_end_frame(self.rollout_data)

        label_upsample_map = upscale_and_get_labels(
            self.data_path, self.rollout_data, self.end_frame
        )
        self.example_weights = torch.nn.functional.normalize(
            torch.FloatTensor(
                [
                    label_upsample_map[
                        tuple(
                            self.rollout_data.expert_controller_no_start_select(
                                str(frame)
                            )
                        )
                    ]
                    for frame in range(FRAMES_TO_SKIP, self.end_frame)
                ],
            ),
            p=1,
            dim=0,
        )

        reward_vector_history = self.rollout_data.reward_vector_history
        agent_params = self.rollout_data.agent_params

        reward_vector_size = reward_vector_history[str(0)].shape[0]

        self.values = torch.zeros(
            (self.end_frame, reward_vector_size), dtype=torch.float
        )
        self.rewards = torch.zeros(
            (self.end_frame, reward_vector_size), dtype=torch.float
        )
        self.advantages = torch.zeros(
            (self.end_frame, reward_vector_size), dtype=torch.float
        )
        self.returns = torch.zeros(
            (self.end_frame, reward_vector_size), dtype=torch.float
        )

        if self.imitation_learning:
            # Imitation learning
            for frame in range(self.end_frame - 1, 0, -1):
                if frame == self.end_frame - 1:
                    self.values[frame] = reward_vector_history[str(frame)]
                else:
                    self.values[frame] = reward_vector_history[str(frame)] + (
                        GAMMA * self.values[frame + 1]
                    )
                self.rewards[frame] = reward_vector_history[str(frame)]
        else:

            lastgaelam = torch.zeros(reward_vector_size, dtype=torch.float)
            for frame in range(self.end_frame - 1, FRAMES_TO_SKIP, -1):
                self.rewards[frame] = reward_vector_history[str(frame)]
                self.values[frame] = values = torch.tensor(
                    agent_params[str(frame)]["value"]
                )

                if frame == self.end_frame - 1:
                    delta = 0.0
                    nextnonterminal = 0.0
                else:
                    nextnonterminal = 1.0
                    next_values = torch.tensor(agent_params[str(frame + 1)]["value"])
                    delta = (
                        torch.tensor(reward_vector_history[str(frame)])
                        + (GAMMA * next_values)
                    ) - values
                self.advantages[frame] = lastgaelam = delta + (
                    GAMMA * GAE_LAMBDA * nextnonterminal * lastgaelam
                )
                # self.returns[frame] = self.advantages[frame] + values

    def __getitem__(self, idx):
        if not self.bootstrapped:
            self.bootstrap()
            self.bootstrapped = True

        frame = idx + FRAMES_TO_SKIP
        while frame >= self.end_frame:
            frame -= 1
        return self.get_frame(frame)

    def get_frame(self, frame):
        if not self.bootstrapped:
            self.bootstrap()
            self.bootstrapped = True

        history_start_frame = frame - 3
        image_list, input_list = zip(
            *[
                self.get_image_and_input_at_frame(history_start_frame + i)
                for i in range(4)
            ]
        )
        assert len(image_list) == 4
        assert len(input_list) == 4
        image_stack = torch.stack(image_list)

        if (
            False
            and self.imitation_learning == False
            and "screen_buffer" in self.rollout_data.agent_params[str(frame)]
        ):
            # print("CHECKING IMAGE STACK")
            assert torch.equal(
                image_stack,
                torch.tensor(
                    self.rollout_data.agent_params[str(frame)]["screen_buffer"]
                ),
            ), f"{image_stack} != {self.rollout_data.agent_params[str(frame)]['screen_buffer']}"

        past_inputs = torch.zeros((3, 8), dtype=torch.float)
        for i in range(3):
            past_inputs[i, :] = torch.FloatTensor(input_list[i])

        if (
            False
            and self.imitation_learning == False
            and "controller_buffer" in self.rollout_data.agent_params[str(frame)]
        ):
            assert torch.equal(
                past_inputs,
                torch.tensor(
                    self.rollout_data.agent_params[str(frame)]["controller_buffer"]
                ),
            ), f"{past_inputs} != {self.rollout_data.agent_params[str(frame)]['controller_buffer']}"

        value = self.values[frame]

        past_rewards = torch.zeros((3, len(self.rewards[frame])), dtype=torch.float)
        # print(past_rewards.shape, self.rewards.shape, frame)
        for i in range(3):
            past_rewards[i] = self.rewards[history_start_frame + i]

        if self.imitation_learning:
            log_prob = 0.0
        else:
            log_prob = self.rollout_data.agent_params[str(frame)]["log_prob"]

        return (
            image_stack,
            value.to(dtype=torch.float32),
            past_inputs,
            past_rewards,
            torch.IntTensor(input_list[-1]),
            torch.tensor(log_prob, dtype=torch.float32),
            self.advantages[frame],
        )
