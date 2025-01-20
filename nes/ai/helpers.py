from collections import Counter
from pathlib import Path
import shutil
import torch
from fastai.vision.all import *
import fastai
import json
import numpy as np

import shelve

def upscale_and_get_labels():
    expert_controller = shelve.open("expert_controller.shelve")

    seen_values = {}

    def get_label(x):
        data_frame_str = str(x).split("/")[-1].split(".")[-2]
        controller_array = expert_controller[data_frame_str]
        return json.dumps(controller_array.tolist())
        controller_int_value = 0
        for i, value in enumerate(controller_array):
            controller_int_value += value * (2 ** i)
        #print(controller_array, controller_int_value)
        return controller_int_value
        if controller_int_value not in seen_values:
            seen_values[controller_int_value] = len(seen_values)+1
        return seen_values[controller_int_value]

    path = "./expert_images"
    image_files = get_image_files(path)
    labels = [get_label(item) for item in image_files]
    file_label_map = {}
    for i in range(len(labels)):
        filename = str(image_files[i]).split("/")[-1]
        file_label_map[filename] = labels[i]

    mark_for_deletion = []
    for i in range(len(labels)):
        # Remove the start button press from the dataset.  We'll press this manually
        if labels[i] == '[false, false, false, true, false, false, false, false]':
            mark_for_deletion.append(i)
    image_files = [item for i, item in enumerate(image_files) if i not in mark_for_deletion]
    labels = [item for i, item in enumerate(labels) if i not in mark_for_deletion]

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
    print("UPSAMPLE FACTORS", upsample_factors)
    while min(upsample_factors) > 8:
        upsample_factors = [factor // 2 for factor in upsample_factors]
        print("UPSAMPLE FACTORS", upsample_factors)
    print("UPSAMPLE FACTORS (done)", upsample_factors)

    image_files_upscaled = []
    for i, item in enumerate(image_files):
        label = labels[i]
        for j in range(upsample_factors[label_names.index(label)]):
            image_files_upscaled.append(item)

    return image_files_upscaled, file_label_map
