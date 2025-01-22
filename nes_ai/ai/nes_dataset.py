import json
import shelve

import torch
from PIL import Image
from torchvision import transforms

from nes_ai.ai.helpers import upscale_and_get_labels

DEFAULT_TRANSFORM = transforms.Compose(
    [
        transforms.ToTensor(),
        transforms.Resize((224, 224)),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]
)


class NESDataset(torch.utils.data.Dataset):
    def __init__(self, data_path, train: bool):
        (
            self.image_files,
            self.file_to_frame,
            image_files_upscaled,
            self.file_label_map,
            label_upsample_map,
        ) = upscale_and_get_labels(data_path)
        self.example_weights = torch.nn.functional.normalize(
            torch.FloatTensor(
                [
                    label_upsample_map[self.file_label_map[f.name]]
                    for f in self.image_files
                ]
            ),
            p=1,
            dim=0,
        )
        print(self.example_weights)
        print(self.example_weights.sum())
        self.train = train
        self.max_frame = max(self.file_to_frame.values())
        self.reward_vector_history = None

    def __len__(self):
        return len(self.image_files)

    def get_image_and_input_at_frame(self, frame):
        image_filename = self.image_files[frame]
        # print("frame to file", frame, image_filename)
        image = Image.open(image_filename)
        image = DEFAULT_TRANSFORM(image)
        # print("Replay buffer image",frame, image.mean())
        return image, self.file_label_map[self.image_files[frame].name]

    def __getitem__(self, idx):
        if self.reward_vector_history is None:
            self.reward_vector_history = shelve.open("reward_vector.shelve", flag="r")

        frame = self.file_to_frame[self.image_files[idx]]
        while frame + 4 > self.max_frame:
            frame -= 1
        current_frame = frame + 4
        image_list, input_list = zip(
            *[self.get_image_and_input_at_frame(frame + i) for i in range(4)]
        )
        assert len(image_list) == 4
        assert len(input_list) == 4
        image_stack = torch.stack(image_list)
        past_inputs = torch.zeros((3, 8), dtype=torch.float)
        for i in range(3):
            past_inputs[i, :] = torch.FloatTensor(input_list[i])

        # The reward R(s,a) at time t is R(s'): reward_vector_history at t+1
        if current_frame == self.max_frame:
            value = torch.zeros_like(
                self.reward_vector_history[str(current_frame)], dtype=torch.float
            )
        else:
            value = torch.clone(self.reward_vector_history[str(current_frame + 1)])
        gamma = 1.0
        for future_step in range(1, 600):
            gamma *= 0.995
            if current_frame + future_step + 1 > self.max_frame:
                break
            value += (
                gamma * self.reward_vector_history[str(current_frame + future_step + 1)]
            )

        past_rewards = torch.zeros(
            (3, len(self.reward_vector_history[str(frame)])), dtype=torch.float
        )
        for i in range(3):
            past_rewards[i, :] = self.reward_vector_history[str(frame + i + 1)]

        return (
            image_stack,
            value,
            past_inputs,
            past_rewards,
            torch.IntTensor(input_list[-1]),
        )
