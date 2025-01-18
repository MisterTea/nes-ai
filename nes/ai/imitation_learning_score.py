import json
from pathlib import Path
import shutil
import torch
from fastai.vision.all import *
import fastai
from PIL import Image
import numpy as np

import shelve
expert_controller = shelve.open("expert_controller.shelve")

labels = ['[false, false, false, false, false, false, false, false]', '[false, false, false, false, false, false, false, true]', '[false, false, false, true, false, false, false, false]', '[false, true, false, false, false, false, false, true]', '[true, true, false, false, false, false, false, true]']

seen_values = {}

def get_label(x):
    data_frame_str = x.split(".")[0]
    controller_array = expert_controller[data_frame_str]
    controller_int_value = 0
    for i, value in enumerate(controller_array):
        controller_int_value += value * (2 ** i)
    #print(x, controller_int_value)
    if controller_int_value not in seen_values:
        seen_values[controller_int_value] = len(seen_values)+1
    return seen_values[controller_int_value]

path = "./expert_images"
dls = ImageDataLoaders.from_name_func(
    path, get_image_files(path), valid_pct=0.2,
    label_func=get_label, item_tfms=Resize(224))

# if a string is passed into the model argument, it will now use timm (if it is installed)
learn = vision_learner(dls, 'vit_tiny_patch16_224', metrics=error_rate, cbs=[
        fastai.callback.tracker.SaveModelCallback(monitor='error_rate', fname='best_model')
    ])
learn.path = Path('./fastai_output')

learn.load("final_model")
print(learn.path)
print(learn.dls.vocab)

learn.export("final_model.pkl")

l = load_learner("fastai_output/final_model.pkl")
print(l)
img = np.array(Image.open("expert_images/264.png"))
x = l.predict(img)
print(x)

def score_image(img):
    x = l.predict(np.array(img))
    label = labels[int(x[0])-1]
    print(label)
    return np.array(json.loads(label))

print(score_image(img))