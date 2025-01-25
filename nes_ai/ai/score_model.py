from __future__ import annotations

from pathlib import Path

import torch
from torch.distributions.categorical import Categorical

from nes_ai.ai.timm_imitation_learning import LitClassification
from nes_ai.ai.timm_rl import LitPPO

inference_model = None


def get_i():
    return inference_model


def score(model_path: Path, images, controller_buffer, reward_history, data_frame):
    global inference_model
    if inference_model is None:
        if True or "rl_model" in str(model_path):
            inference_model = LitPPO.load_from_checkpoint(model_path)
        else:
            inference_model = LitClassification.load_from_checkpoint(model_path)
        inference_model.train()  # We score the model in train() mode because the score is used for further training
    with torch.no_grad():
        moved_images = images.unsqueeze(0).to(device=inference_model.device)
        value = inference_model.critic(
            moved_images,
            controller_buffer.unsqueeze(0).to(device=inference_model.device),
            reward_history.unsqueeze(0).to(device=inference_model.device),
        )
        action, action_log_prob, entropy = inference_model.actor.get_action(
            moved_images,
            controller_buffer.unsqueeze(0).to(device=inference_model.device),
            None,
        )
        return (
            action.squeeze(0),
            action_log_prob.squeeze(0),
            entropy.squeeze(0),
            value.squeeze(0),
        )

        label_logits, value = inference_model(
            images.unsqueeze(0),
            controller_buffer.unsqueeze(0),
            reward_history.unsqueeze(0),
        )
        print("LOGITS", label_logits, label_logits.min(), label_logits.max())
        label_logits = label_logits.squeeze(0)
        value = value.squeeze(0)
        probs = Categorical(logits=label_logits)
        # label_probs = torch.nn.functional.softmax(label_logits)
        drawn_action_index = probs.sample()
        # print(label_probs)
        # drawn_action_index = torch.argmax(label_probs).item()
        # print(drawn_action_index)
        # print(inference_model.int_label_map)
        # drawn_action = inference_model.int_label_map[drawn_action_index.item()]
        drawn_action = inference_model.actor.convert_index_to_input_array(
            drawn_action_index
        )

        return drawn_action, probs.log_prob(drawn_action_index), probs.entropy(), value
