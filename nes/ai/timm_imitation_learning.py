import json
import math
import shutil
import time
from collections import Counter
from pathlib import Path

import numpy as np
import pytorch_lightning as pl
import timm
import torch
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.loggers import TensorBoardLogger
from torcheval.metrics import MulticlassAccuracy
from torchvision import datasets, transforms

from nes.ai.helpers import upscale_and_get_labels
from nes.ai.nes_dataset import NESDataset

avail_pretrained_models = timm.list_models(pretrained=True)
print(len(avail_pretrained_models), avail_pretrained_models)

BATCH_SIZE = 64


def _linear_block(in_features, out_features):
    return [
        torch.nn.Linear(in_features, out_features),
        torch.nn.LeakyReLU(),
        torch.nn.BatchNorm1d(out_features),
    ]


class LitClassification(pl.LightningModule):
    def __init__(self):
        super().__init__()
        self.trunk = timm.create_model(
            "vit_tiny_patch16_224", pretrained=True, num_classes=0
        )
        self.head = torch.nn.Sequential(
            *_linear_block((self.trunk.num_features * 4) + (8*3), 1024),
            *_linear_block(1024, 1024),
            torch.nn.Linear(1024, 5),
        )
        self.loss_fn = torch.nn.CrossEntropyLoss()

    def forward(self, images, past_inputs):
        assert images.shape[1:] == (4, 3, 224, 224), f"{images.shape}"
        assert past_inputs.shape[1:] == (3, 8), f"{past_inputs.shape}"
        trunk_output = torch.cat(
            [self.trunk(images[:, x, :, :, :]) for x in range(4)], dim=1
        )
        trunk_output = torch.cat((trunk_output, past_inputs.reshape(-1, 3*8)), dim=1)
        outputs = self.head(trunk_output)
        # print("***")
        # print(targets)
        # print(outputs)
        # print(targets.shape)
        # print(outputs.shape)
        return outputs

    def training_step(self, batch):
        images, past_inputs, targets = batch

        outputs = self.forward(images, past_inputs)
        loss = self.loss_fn(outputs, targets)
        self.log("train_loss", loss, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):

        # print(batch)
        images, past_inputs, targets = batch

        outputs = self.forward(images, past_inputs)

        loss = self.loss_fn(outputs, targets)
        self.log("val_loss", loss, prog_bar=True)

        metric = MulticlassAccuracy()
        metric.update(outputs, targets)
        accuracy = metric.compute()
        self.log("val_acc", accuracy, prog_bar=True)

        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=0.00001 * BATCH_SIZE)
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": torch.optim.lr_scheduler.ReduceLROnPlateau(
                    optimizer, mode="min", factor=math.sqrt(0.1), patience=2
                ),
                "monitor": "train_loss",
                "frequency": 1,
                # If "monitor" references validation metrics, then "frequency" should be set to a
                # multiple of "trainer.check_val_every_n_epoch".
            },
        }


class ClassificationData(pl.LightningDataModule):

    def train_dataloader(self):
        train_dataset = NESDataset(train=True)
        return torch.utils.data.DataLoader(
            train_dataset, batch_size=BATCH_SIZE, num_workers=10,# shuffle=True,
            sampler=torch.utils.data.WeightedRandomSampler(
                train_dataset.example_weights, BATCH_SIZE * 100, replacement=True
            ),
        )

    def val_dataloader(self):
        val_dataset = NESDataset(train=False)
        return torch.utils.data.DataLoader(
            val_dataset,
            batch_size=BATCH_SIZE,
            num_workers=10,
            #sampler=torch.utils.data.RandomSampler(val_dataset, num_samples=100),
        )

hashes = {}
def find_image_input(image, data_frame):
    import imagehash
    from PIL import Image
    global hashes
    if len(hashes) == 0:
        # Populate hashes
        for path in Path("expert_images").glob("*.png"):
            hashes[imagehash.phash(Image.open(path))] = path
    input_hash = imagehash.phash(image)

    replay_image_filename = "expert_images/" + str(data_frame) + ".png"
    replay_image_hash = imagehash.phash(Image.open(replay_image_filename))
    #assert input_hash == replay_image_hash, f"{input_hash} != {replay_image_hash}"

    if input_hash in hashes:
        #print("MATCH", hashes[input_hash])
        return hashes[input_hash]
    return None

inference_model = None
inference_dataset = None
def score(images, controller_buffer, ground_truth_controller, data_frame):
    global inference_model
    global inference_dataset
    print("Scoring",data_frame)
    if inference_model is None:
        inference_model = LitClassification.load_from_checkpoint('timm_il_models/best_model-v10.ckpt').cpu()
        inference_model.eval()
        inference_dataset = NESDataset(train=False)
        lim = inference_dataset.label_int_map
        int_label_map = {}
        for k, v in lim.items():
            int_label_map[v] = json.loads(k)
        inference_model.int_label_map = int_label_map
    with torch.no_grad():
        label_logits = inference_model(images.unsqueeze(0), controller_buffer.unsqueeze(0)).squeeze(0)
        #print(label_logits)
        label_probs = torch.nn.functional.softmax(label_logits)
        #print(label_probs)
        highest_prob = torch.argmax(label_probs).item()
        #print(highest_prob)
        #print(inference_model.int_label_map)
        label = inference_model.int_label_map[highest_prob]

        if False: # Check against ground truth
            image_stack, past_inputs, label_int = inference_dataset[int(data_frame) - 3]
            assert torch.equal(past_inputs,controller_buffer), f"{past_inputs} != {controller_buffer}"
            if not torch.equal(image_stack,images):
                print(image_stack[3].mean(), images[3].mean())
                assert torch.equal(image_stack[0], images[0]), f"{image_stack[0]} != {images[0]}"
                assert torch.equal(image_stack[1], images[0]), f"{image_stack[1]} != {images[1]}"
                assert torch.equal(image_stack[2], images[0]), f"{image_stack[2]} != {images[2]}"
                assert torch.equal(image_stack[3], images[0]), f"{image_stack[3]} != {images[3]}"

        if not torch.equal(torch.Tensor(label), torch.Tensor(ground_truth_controller)):
            print("WRONG", label, ground_truth_controller)
            #return list(ground_truth_controller)
        #print(label)
        return label

if __name__ == "__main__":
    data = ClassificationData()
    model = LitClassification()
    model.int_label_map = data.train_dataloader().dataset.int_label_map
    trainer = pl.Trainer(
        max_epochs=2000,
        accelerator="auto",
        logger=TensorBoardLogger("logs/", name="timm_il_logs"),
        callbacks=[
            LearningRateMonitor(logging_interval="step"),
            EarlyStopping(
                monitor="val_loss",
                min_delta=0.0001,
                mode="min",
                patience=100,
                verbose=True,
            ),
            ModelCheckpoint(
                dirpath="timm_il_models",
                monitor="val_loss",
                mode="min",
                save_top_k=1,
                filename="best_model",
            ),
        ],
    )
    trainer.fit(model, data)
