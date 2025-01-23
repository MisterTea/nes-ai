from __future__ import annotations

import torch
from torch.distributions.categorical import Categorical

from nes_ai.ai.timm_imitation_learning import LitClassification

inference_model = None


def score(images, controller_buffer, reward_history, data_frame):
    global inference_model
    print("Scoring", data_frame)
    if inference_model is None:
        inference_model = LitClassification.load_from_checkpoint(
            "timm_il_models/best_model-v3.ckpt"
        ).cpu()
        inference_model.eval()
    with torch.no_grad():
        label_logits, value = inference_model(
            images.unsqueeze(0),
            controller_buffer.unsqueeze(0),
            reward_history.unsqueeze(0),
        )
        label_logits = label_logits.squeeze(0)
        value = value.squeeze(0)
        print("label_logits", label_logits)
        probs = Categorical(logits=label_logits)
        # label_probs = torch.nn.functional.softmax(label_logits)
        drawn_action_index = probs.sample()
        # print(label_probs)
        # drawn_action_index = torch.argmax(label_probs).item()
        # print(drawn_action_index)
        # print(inference_model.int_label_map)
        # drawn_action = inference_model.int_label_map[drawn_action_index.item()]
        drawn_action = inference_model.actor.convert_index_to_input_array(
            drawn_action_index.item()
        )

        return drawn_action, probs.log_prob(drawn_action_index), probs.entropy(), value
