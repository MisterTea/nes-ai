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

GAMMA = 0.995
GAE_LAMBDA = 0.95

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

        self.bootstrapped = False

    def __len__(self):
        return len(self.image_files)

    def get_image_and_input_at_frame(self, frame):
        image_filename = self.image_files[frame - 200]
        # print("frame to file", frame, image_filename)
        image = Image.open(image_filename)
        image = DEFAULT_TRANSFORM(image)
        # print("Replay buffer image",frame, image.mean())
        #print("Getting image and input at frame", frame, image_filename, self.file_label_map[image_filename.name])
        return image, self.file_label_map[image_filename.name]

    def bootstrap(self):
        reward_vector_history = shelve.open("reward_vector.shelve", flag="r")
        agent_params = shelve.open("agent_params.shelve")

        reward_vector_size = reward_vector_history[str(0)].shape[0]

        self.values = torch.zeros((self.max_frame + 1, reward_vector_size), dtype=torch.float)
        self.rewards = torch.zeros((self.max_frame + 1, reward_vector_size), dtype=torch.float)
        self.advantages = torch.zeros((self.max_frame + 1, reward_vector_size), dtype=torch.float)
        self.returns = torch.zeros((self.max_frame + 1, reward_vector_size), dtype=torch.float)

        if True or len(agent_params) == 0:
            # Imitation learning
            for frame in range(self.max_frame, 0, -1):
                if frame == self.max_frame:
                    self.values[frame] = reward_vector_history[str(frame)]
                else:
                    self.values[frame] = (
                        reward_vector_history[str(frame)]
                        + (GAMMA * self.values[frame+1])
                    )
                self.rewards[frame] = reward_vector_history[str(frame)]
        else:

            lastgaelam = torch.zeros(reward_vector_size, dtype=torch.float)
            for frame in range(self.max_frame, 0, -1):
                self.rewards[frame] = reward_vector_history[str(frame)]
                self.values[frame] = values = agent_params[str(frame)]['value']
                next_values = agent_params[str(frame+1)]['value']

                if frame == self.max_frame:
                    delta = 0.0
                    nextnonterminal = 0.0
                else:
                    nextnonterminal = 1.0
                    delta = (
                        (reward_vector_history[str(frame)]
                        + (GAMMA * next_values))
                        - values
                    )
                self.advantages[frame] = lastgaelam = delta + (GAMMA * GAE_LAMBDA * nextnonterminal * lastgaelam)
                self.returns[frame] = self.advantages[frame] + values

    def __getitem__(self, idx):
        frame = self.file_to_frame[self.image_files[idx]]
        while frame > self.max_frame:
            frame -= 1
        return self.get_frame(frame)
    
    def get_frame(self, frame):
        if not self.bootstrapped:
            self.bootstrap()
            self.bootstrapped = True

        history_start_frame = frame - 3
        image_list, input_list = zip(
            *[self.get_image_and_input_at_frame(history_start_frame + i) for i in range(4)]
        )
        assert len(image_list) == 4
        assert len(input_list) == 4
        image_stack = torch.stack(image_list)
        past_inputs = torch.zeros((3, 8), dtype=torch.float)
        for i in range(3):
            past_inputs[i, :] = torch.FloatTensor(input_list[i])

        value = self.values[frame]

        past_rewards = torch.zeros(
            (3, len(self.rewards[frame])), dtype=torch.float
        )
        #print(past_rewards.shape, self.rewards.shape, frame)
        for i in range(3):
            past_rewards[i] = self.rewards[history_start_frame + i]

        return (
            image_stack,
            value,
            past_inputs,
            past_rewards,
            torch.IntTensor(input_list[-1]),
        )
