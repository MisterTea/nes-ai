from collections import Counter
from pathlib import Path
import shutil
import torch
import fastai
import json
import numpy as np
import copy
import os

import shelve

def upscale_and_get_labels():
    expert_controller = shelve.open("expert_controller.shelve")

    def get_label(x):
        data_frame_str = str(x).split("/")[-1].split(".")[-2]
        controller_array = expert_controller[data_frame_str]
        controller_array[3] = False # Do not press start with AI
        return json.dumps(controller_array.tolist())

    path = "./expert_images"
    image_files = []
    i = 0
    while True:
        if os.path.exists(f"{path}/{i}.png"):
            image_files.append(Path(f"{path}/{i}.png"))
            i += 1
        else:
            break
    labels = [get_label(item) for item in image_files]
    file_label_map = {}
    for i in range(len(labels)):
        file_label_map[image_files[i].name] = labels[i]

    count = Counter(labels)
    print(count)
    label_names, label_freq = [], []
    for key, value in count.items():
        label_names.append(key)
        label_freq.append(value)
    #wgts = [1.0/count[get_label(item)] for item in image_files]
    lcm = np.lcm.reduce(label_freq).item()
    print("LCM", lcm)
    upsample_factors = []
    for freq in label_freq:
        upsample_factors.append(lcm // freq)
    upsample_factors_raw = copy.deepcopy(upsample_factors)
    print("UPSAMPLE FACTORS", upsample_factors)
    while min(upsample_factors) > 8:
        upsample_factors = [factor // 2 for factor in upsample_factors]
        print("UPSAMPLE FACTORS", upsample_factors)
    print("UPSAMPLE FACTORS (done)", upsample_factors)
    label_upsample_map = {}
    for i in range(len(label_names)):
        label_upsample_map[label_names[i]] = upsample_factors[i]

    image_files_upscaled = []
    for i, item in enumerate(image_files):
        label = labels[i]
        for j in range(upsample_factors[label_names.index(label)]):
            image_files_upscaled.append(item)

    file_to_frame = {}
    for i, item in enumerate(image_files):
        file_to_frame[item] = i
    return image_files, file_to_frame, image_files_upscaled, file_label_map, label_upsample_map
