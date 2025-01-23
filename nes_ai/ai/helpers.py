import copy
import json
import os
import shutil
from collections import Counter
from pathlib import Path

import numpy as np
import torch

from nes_ai.ai.base import SELECT, START


def upscale_and_get_labels(data_path, rollout_data, end_frame):
    print(data_path)

    labels = [
        tuple(rollout_data.expert_controller_no_start_select(str(frame)))
        for frame in range(200, end_frame)
    ]

    count = Counter(labels)
    print(count)
    label_names, label_freq = [], []
    for key, value in count.items():
        label_names.append(key)
        label_freq.append(value)

    lcm = np.lcm.reduce(label_freq).item()
    print("LCM", lcm)
    upsample_factors = []
    for freq in label_freq:
        upsample_factors.append(lcm // freq)
    print("UPSAMPLE FACTORS", upsample_factors)
    while min(upsample_factors) > 8:
        upsample_factors = [factor // 2 for factor in upsample_factors]
        print("UPSAMPLE FACTORS", upsample_factors)
    print("UPSAMPLE FACTORS (done)", upsample_factors)
    label_upsample_map = {}
    for i in range(len(label_names)):
        label_upsample_map[label_names[i]] = upsample_factors[i]

    return label_upsample_map
