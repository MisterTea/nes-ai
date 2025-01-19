from collections import Counter
from pathlib import Path
import shutil
import torch
from fastai.vision.all import *
import fastai
import json
import numpy as np

import shelve
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
while min(upsample_factors) > 4:
    upsample_factors = [factor // 2 for factor in upsample_factors]
    print("UPSAMPLE FACTORS", upsample_factors)
print("UPSAMPLE FACTORS (done)", upsample_factors)

image_files_upscaled = []
for i, item in enumerate(image_files):
    label = labels[i]
    for j in range(upsample_factors[label_names.index(label)]):
        image_files_upscaled.append(item)

dls = ImageDataLoaders.from_name_func(
    path, image_files_upscaled, valid_pct=0.2,
    label_func=get_label, item_tfms=Resize(224))

# if a string is passed into the model argument, it will now use timm (if it is installed)
learn = vision_learner(dls, 'vit_tiny_patch16_224', metrics=error_rate, cbs=[
        fastai.callback.tracker.SaveModelCallback(monitor='error_rate', fname='best_model', with_opt=True),
        fastai.callback.tracker.EarlyStoppingCallback (monitor='error_rate', comp=None, min_delta=0.0,
                        patience=10, reset_on_fit=True),
    ])
shutil.rmtree(Path('./fastai_output'), ignore_errors=True)
learn.path = Path('./fastai_output')

learn.fine_tune(200000, freeze_epochs=10)
print(learn.path)
print(learn.dls.vocab)

labels = ['[false, false, false, false, false, false, false, false]', '[false, false, false, false, false, false, false, true]', '[false, true, false, false, false, false, false, true]', '[true, true, false, false, false, false, false, true]']

assert json.dumps(list(learn.dls.vocab)) == json.dumps(labels)

learn.save("final_model")