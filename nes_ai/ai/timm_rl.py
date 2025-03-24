from __future__ import annotations

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
from pydantic import BaseModel
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.loggers import TensorBoardLogger
from torch.distributions.categorical import Categorical
from torcheval.metrics import MulticlassAccuracy
from torchvision import datasets, transforms

from nes_ai.ai.base import Actor, Critic, RewardMap
from nes_ai.ai.helpers import upscale_and_get_labels
from nes_ai.ai.nes_dataset import NESDataset

# avail_pretrained_models = timm.list_models(pretrained=True)
# print(len(avail_pretrained_models), avail_pretrained_models)

BATCH_STEP_SIZE = 32
BATCH_SIZE = 32
REWARD_VECTOR_SIZE = 8
CLIP_COEFFICIENT = 0.1
ENTROPY_LOSS_SCALE = 0.01


class LitPPO(pl.LightningModule):
    def __init__(self):
        super().__init__()
        self.actor = Actor()
        self.critic = Critic()
        self.actor_loss_fn = torch.nn.CrossEntropyLoss()
        self.reward_loss_fn = torch.nn.SmoothL1Loss()
        self.automatic_optimization = False

    def forward(self, images, past_inputs, past_rewards):
        return self.actor.forward(images, past_inputs), self.critic.forward(
            images, past_inputs, past_rewards
        )

    def training_step(self, batch, batch_idx):
        (
            images,
            recorded_ground_truth_values,
            past_inputs,
            past_rewards,
            action_taken,
            recorded_action_log_prob,
            advantages,
        ) = batch

        torch.autograd.set_detect_anomaly(True)
        actor_opt, critic_opt = self.optimizers()

        advantages_accum = RewardMap.combine_reward_vector(advantages)

        if True:  # Advantage normalization
            advantages_accum = (advantages_accum - advantages_accum.mean()) / (
                advantages_accum.std() + 1e-8
            )

        returns = advantages + recorded_ground_truth_values

        predicted_values = self.critic.forward(
            images,
            past_inputs,
            past_rewards,
        )

        # Critic loss
        if True:
            # Clip value loss
            v_loss_unclipped = torch.nn.SmoothL1Loss()(predicted_values, returns)
            v_clipped = recorded_ground_truth_values + torch.clamp(
                predicted_values - recorded_ground_truth_values,
                -CLIP_COEFFICIENT,
                CLIP_COEFFICIENT,
            )
            v_loss_clipped = torch.nn.SmoothL1Loss()(v_clipped, returns)
            v_loss_max = torch.maximum(v_loss_unclipped, v_loss_clipped)
            critic_loss = 0.5 * v_loss_max.mean()
        else:
            critic_loss = torch.nn.SmoothL1Loss()(predicted_values, returns)

        self.log("critic_train_loss", critic_loss, prog_bar=True)

        critic_opt.zero_grad()
        self.manual_backward(critic_loss)
        critic_opt.step()

        action, action_log_prob, entropy = self.actor.get_action(
            images, past_inputs, action_taken
        )
        assert torch.equal(action, action_taken)
        assert action_log_prob.shape == recorded_action_log_prob.shape

        from nes_ai.ai.score_model import get_i, score

        # _action, _action_lp, _entropy, _value = score(
        #     "timm_il_models/best_model-v10.ckpt",
        #     images[0].cpu(),
        #     past_inputs[0].cpu(),
        #     past_rewards[0].cpu(),
        #     None,
        # )
        # a3, alp3, e3 = get_i().actor.get_action(images, past_inputs, action_taken)
        # print(a3)
        # print(alp3)
        # print(e3)
        # print(images.squeeze(0).device)
        # print(action_taken)
        # print(recorded_action_log_prob)
        # print(_entropy)
        # print("fart")
        # print(action_log_prob)
        # print(recorded_action_log_prob)
        # alsjdlkasjklj
        # They won't be identical because of batch norm
        # if torch.abs(action_log_prob - recorded_action_log_prob).mean() > 0.1:
        #     print("ACTION LOG PROB MISMATCH")
        #     print(action_log_prob)
        #     print(recorded_action_log_prob)
        #     print(action)
        #     print(action_taken)
        #     print(entropy)
        #     print(advantages)
        #     print(_action)
        #     print(_action_lp)
        #     print("ACTION LOG PROB MISMATCH")
        #     asdlkjasd
        logratio = torch.clamp(action_log_prob - recorded_action_log_prob, -10, 10)
        print("LOGRATIO", logratio)
        ratio = logratio.exp()

        with torch.no_grad():
            # calculate approx_kl http://joschu.net/blog/kl-approx.html
            old_approx_kl = (-logratio).mean()
            approx_kl = ((ratio - 1) - logratio).mean()
            # clipfracs += [
            #     ((ratio - 1.0).abs() > CLIP_COEFFICIENT).float().mean().item()
            # ]
            # print("KL INFO")
            # print(logratio, ratio, approx_kl, old_approx_kl)
            self.log(
                "approx_kl",
                approx_kl,
                prog_bar=True,
                on_step=True,
                on_epoch=True,
                sync_dist=True,
            )

        # Policy loss
        ratio = ratio.unsqueeze(1)
        pg_loss1 = -advantages_accum
        pg_loss2 = -advantages_accum * torch.clamp(
            ratio, 1 - CLIP_COEFFICIENT, 1 + CLIP_COEFFICIENT
        )
        pg_loss = torch.max(pg_loss1, pg_loss2).mean()

        # Entropy loss
        entropy_loss = -1 * entropy.mean()

        # Total actor loss
        actor_loss = pg_loss + (ENTROPY_LOSS_SCALE * entropy_loss)

        self.log("actor_train_loss", actor_loss, prog_bar=True)

        actor_opt.zero_grad()
        self.manual_backward(actor_loss)
        actor_opt.step()

        self.log("train_loss", actor_loss + critic_loss, prog_bar=False)

        actor_sch, critic_sch = self.lr_schedulers()

        actor_sch.step(self.trainer.callback_metrics["actor_train_loss"])
        critic_sch.step(self.trainer.callback_metrics["critic_train_loss"])

        self.log("actor_lr", actor_sch.get_last_lr()[0], prog_bar=True)
        self.log("critic_lr", critic_sch.get_last_lr()[0], prog_bar=True)

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.actor.parameters(), lr=0.0001 * BATCH_SIZE)
        actor_opt = {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": torch.optim.lr_scheduler.ReduceLROnPlateau(
                    optimizer, mode="min", factor=math.sqrt(0.1), patience=2000
                ),
                # "monitor": "actor_train_loss",
                # "frequency": 1,
                # If "monitor" references validation metrics, then "frequency" should be set to a
                # multiple of "trainer.check_val_every_n_epoch".
            },
        }
        optimizer = torch.optim.AdamW(self.critic.parameters(), lr=0.00001 * BATCH_SIZE)
        critic_opt = {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": torch.optim.lr_scheduler.ReduceLROnPlateau(
                    optimizer, mode="min", factor=math.sqrt(0.1), patience=2000
                ),
                # "monitor": "critic_train_loss",
                # "frequency": 1,
                # If "monitor" references validation metrics, then "frequency" should be set to a
                # multiple of "trainer.check_val_every_n_epoch".
            },
        }
        return actor_opt, critic_opt


class ClassificationData(pl.LightningDataModule):
    def __init__(self, data_path: Path, imitation_learning=True):
        super().__init__()
        self.imitation_learning = imitation_learning
        self.data_path = data_path

    def train_dataloader(self):
        # Spin up dummy set to compute example weights
        dummy_train_dataset = NESDataset(
            self.data_path, train=True, imitation_learning=self.imitation_learning
        )
        dummy_train_dataset.bootstrap()
        example_weights = dummy_train_dataset.example_weights
        assert len(example_weights) == len(
            dummy_train_dataset
        ), f"{len(example_weights)} != {len(dummy_train_dataset)}"

        train_dataset = NESDataset(
            self.data_path, train=True, imitation_learning=self.imitation_learning
        )
        # return torch.utils.data.DataLoader(
        #     train_dataset,
        #     batch_size=BATCH_SIZE,
        #     # persistent_workers=True,
        #     num_workers=0,  # shuffle=True,
        #     sampler=torch.utils.data.WeightedRandomSampler(
        #         example_weights, BATCH_STEP_SIZE * 100, replacement=True
        #     ),
        # )
        return torch.utils.data.DataLoader(
            train_dataset,
            batch_size=BATCH_SIZE,
            # persistent_workers=True,
            num_workers=10,
            shuffle=True,
        )

    def val_dataloader(self):
        val_dataset = NESDataset(
            self.data_path, train=False, imitation_learning=self.imitation_learning
        )
        return torch.utils.data.DataLoader(
            val_dataset,
            batch_size=BATCH_SIZE,
            # persistent_workers=True,
            num_workers=0,
            # sampler=torch.utils.data.RandomSampler(val_dataset, num_samples=100),
        )


import click


def train_rl(data_path: Path, checkpoint_path: Path | None = None):
    data = ClassificationData(data_path, imitation_learning=False)
    if checkpoint_path is None:
        model = LitPPO()
    else:
        print("Loading from checkpoint", checkpoint_path)
        model = LitPPO.load_from_checkpoint(checkpoint_path)
    trainer = pl.Trainer(
        max_epochs=1,
        accelerator="auto",
        logger=TensorBoardLogger("logs/", name="timm_rl_logs"),
        callbacks=[
            LearningRateMonitor(logging_interval="step"),
            # EarlyStopping(
            #     monitor="approx_kl_epoch",
            #     divergence_threshold=0.05,
            #     verbose=True,
            # ),
            ModelCheckpoint(
                dirpath="timm_rl_models",
                # monitor="val_loss",
                # mode="min",
                save_top_k=1,
                filename="best_model",
            ),
        ],
    )
    trainer.fit(model, data)


@click.command()
@click.option("--checkpoint-path")
def main(checkpoint_path: Path | None = None):
    DATA_PATH = Path("./data/1_1_rl")
    return train_rl(data_path=DATA_PATH, checkpoint_path=checkpoint_path)


if __name__ == "__main__":
    main()
