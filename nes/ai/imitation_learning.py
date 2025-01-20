from collections import Counter
from pathlib import Path
import shutil
import torch
from fastai.vision.all import *
import fastai
import json
import numpy as np

import shelve

from nes.ai.helpers import upscale_and_get_labels

def train(image_files_upscaled, file_label_map):
    path = "./expert_images"
    dls = ImageDataLoaders.from_name_func(
        path, image_files_upscaled, valid_pct=0.2,
        label_func=lambda x: file_label_map[x], item_tfms=Resize(224))

    # if a string is passed into the model argument, it will now use timm (if it is installed)
    learn = vision_learner(dls, 'vit_tiny_patch16_224', metrics=error_rate, cbs=[
            fastai.callback.tracker.SaveModelCallback(monitor='error_rate', fname='best_model', with_opt=True),
            fastai.callback.tracker.EarlyStoppingCallback (monitor='error_rate', comp=None, min_delta=0.0,
                            patience=10, reset_on_fit=True),
        ])
    shutil.rmtree(Path('./fastai_output'), ignore_errors=True)
    learn.path = Path('./fastai_output')

    learn.fine_tune(200000, freeze_epochs=1)
    print(learn.path)
    print(learn.dls.vocab)

    labels = ['[false, false, false, false, false, false, false, false]', '[false, false, false, false, false, false, false, true]', '[false, true, false, false, false, false, false, true]', '[true, true, false, false, false, false, false, true]']

    assert json.dumps(list(learn.dls.vocab)) == json.dumps(labels)

    learn.save("final_model")

if __name__ == "__main__":
    image_files_upscaled, file_label_map = upscale_and_get_labels()
    train(image_files_upscaled, file_label_map)