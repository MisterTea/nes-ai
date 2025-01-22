import shelve

import numpy as np
import torch
from PIL import Image

from nes_ai.ai.timm_imitation_learning import compute_reward_map


class AiHandler:
    def __init__(self):
        self.expert_controller = shelve.open("expert_controller.shelve")
        self.expert_input = shelve.open("expert_input.shelve")
        self.reward_map_history = shelve.open("reward_map.shelve")
        self.reward_vector_history = shelve.open("reward_vector.shelve")
        self.agent_params = shelve.open("agent_params.shelve")

        self.controller_buffer = torch.zeros((3, 8), dtype=torch.float)
        self.screen_buffer = torch.zeros((4, 3, 224, 224), dtype=torch.float)
        self.last_reward_map = None
        self.reward_map = None

    def shutdown(self):
        self.expert_controller.close()
        self.expert_input.close()
        self.reward_map_history.close()
        self.reward_vector_history.close()
        self.agent_params.close()

    def update(self, frame, controller1, ram, screen_buffer_image):
        self.last_reward_map = self.reward_map
        reward_map, reward_vector = compute_reward_map(
            self.last_reward_map, torch.from_numpy(ram).int()
        )
        print("REWARD", reward_map, reward_vector)

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

            if False:
                if frame == 0:
                    print("RESETTING")
                    import shutil

                    shutil.rmtree("expert_images", ignore_errors=True)
                    import os

                    os.mkdir("expert_images")
                    self.expert_controller.clear()
                    self.expert_input.clear()
                    self.reward_map_history.clear()
                    self.reward_vector_history.clear()
                    self.agent_params.clear()

                # Data Collect
                self.expert_controller[str(frame)] = np.array(controller1.is_pressed)
                self.reward_map_history[str(frame)] = reward_map
                self.reward_vector_history[str(frame)] = reward_vector
                screen_buffer_image.save("expert_images/" + str(frame) + ".png")
                if frame % 60 == 0:
                    print("SAVING")
                    self.expert_controller.sync()
                    self.expert_input.sync()
                    self.reward_map_history.sync()
                    self.reward_vector_history.sync()
                    self.agent_params.sync()

            if True:
                # Replay
                controller = self.expert_controller[str(frame)]
                controller1.set_state(controller)
                self.reward_map_history[str(frame)] = reward_map
                self.reward_vector_history[str(frame)] = reward_vector

            if False:
                # Score model
                from nes_ai.ai.imitation_learning_score import score_image

                h = 240 if self.v_overscan else 224
                w = 256 if self.h_overscan else 240
                sb = np.zeros((w, h), dtype=np.uint32)
                self.ppu.copy_screen_buffer_to(
                    sb, v_overscan=self.v_overscan, h_overscan=self.h_overscan
                )
                sb = (
                    sb.view(dtype=np.uint8)
                    .reshape((w, h, 4))[:, :, np.array([2, 1, 0])]
                    .swapaxes(0, 1)
                )
                image = Image.fromarray(sb)
                assert image.size == (w, h)
                image = image.resize((224, 224))
                new_state = score_image(image)
                controller1.update()
                if not controller1.is_any_pressed():
                    controller1.set_state(new_state)

            if False:
                if frame > 200:
                    # Score model timm
                    from nes_ai.ai.timm_imitation_learning import (
                        REWARD_VECTOR_SIZE,
                        score,
                    )

                    print(self.screen_buffer.shape, image.shape)
                    ground_truth_controller = self.expert_controller[str(frame)]
                    reward_history = torch.zeros(
                        (3, REWARD_VECTOR_SIZE), dtype=torch.float
                    )
                    for x in range(3):
                        reward_history[x, :] = torch.FloatTensor(
                            self.reward_vector_history[str((frame - 3) + x)]
                        )
                    action, log_prob, entropy, value = score(
                        self.screen_buffer,
                        self.controller_buffer,
                        ground_truth_controller,
                        reward_history,
                        str(frame),
                    )
                    self.agent_params[str(frame)] = {
                        "action": action,
                        "log_prob": log_prob,
                        "entropy": entropy,
                        "value": value,
                    }
                    controller1.update()
                    if not controller1.is_any_pressed():
                        print(action)
                        controller1.set_state(action)
                else:
                    # Use expert for intro
                    controller = self.expert_controller[str(frame)]
                    controller1.set_state(controller)

            for buffer_i in range(2):
                self.controller_buffer[buffer_i, :] = self.controller_buffer[
                    buffer_i + 1, :
                ]
            self.controller_buffer[2, :] = torch.FloatTensor(controller1.is_pressed)
