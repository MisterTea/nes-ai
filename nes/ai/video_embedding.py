import json
import os
import shelve
import shutil
from collections import Counter
from pathlib import Path

import av
import fastai
import numpy as np
import torch
from fastai.vision.all import *
from huggingface_hub import hf_hub_download
from transformers import EarlyStoppingCallback as TFEarlyStoppingCallback
from transformers import VivitConfig, VivitForVideoClassification, VivitImageProcessor

torch.multiprocessing.set_start_method("spawn")


def main():
    np.random.seed(0)

    expert_controller = shelve.open("expert_controller.shelve")

    seen_values = {}

    def get_label(x):
        data_frame_str = os.path.basename(str(x)).split(".")[-2]
        controller_array = expert_controller[data_frame_str]
        return json.dumps(controller_array.tolist())
        controller_int_value = 0
        for i, value in enumerate(controller_array):
            controller_int_value += value * (2**i)
        # print(controller_array, controller_int_value)
        return controller_int_value
        if controller_int_value not in seen_values:
            seen_values[controller_int_value] = len(seen_values) + 1
        return seen_values[controller_int_value]

    path = "expert_images"
    image_files = get_image_files(path)
    labels = [get_label(item) for item in image_files]
    print("NUM", len(image_files), len(labels))
    mark_for_deletion = []
    for i in range(len(labels)):
        # Remove the start button press from the dataset.  We'll press this manually
        if labels[i] == "[false, false, false, true, false, false, false, false]":
            mark_for_deletion.append(i)
    image_files = [
        item for i, item in enumerate(image_files) if i not in mark_for_deletion
    ]
    labels = [item for i, item in enumerate(labels) if i not in mark_for_deletion]

    count = Counter(labels)
    print(count)
    label_names, label_freq = [], []
    for key, value in count.items():
        label_names.append(key)
        label_freq.append(value)
    # wgts = [1.0/count[get_label(item)] for item in image_files]
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

    def read_video_pyav(container, indices):
        """
        Decode the video with PyAV decoder.
        Args:
            container (`av.container.input.InputContainer`): PyAV container.
            indices (`List[int]`): List of frame indices to decode.
        Returns:
            result (np.ndarray): np array of decoded frames of shape (num_frames, height, width, 3).
        """
        frames = []
        container.seek(0)
        start_index = indices[0]
        end_index = indices[-1]
        for i, frame in enumerate(container.decode(video=0)):
            if i > end_index:
                break
            if i >= start_index and i in indices:
                frames.append(frame)
        return np.stack([x.to_ndarray(format="rgb24") for x in frames])

    def sample_frame_indices(clip_len, frame_sample_rate, seg_len):
        """
        Sample a given number of frame indices from the video.
        Args:
            clip_len (`int`): Total number of frames to sample.
            frame_sample_rate (`int`): Sample every n-th frame.
            seg_len (`int`): Maximum allowed index of sample's last frame.
        Returns:
            indices (`List[int]`): List of sampled frame indices
        """
        converted_len = int(clip_len * frame_sample_rate)
        end_idx = np.random.randint(converted_len, seg_len)
        start_idx = end_idx - converted_len
        indices = np.linspace(start_idx, end_idx, num=clip_len)
        indices = np.clip(indices, start_idx, end_idx - 1).astype(np.int64)
        return indices

    # video clip consists of 300 frames (10 seconds at 30 FPS)
    file_path = hf_hub_download(
        repo_id="nielsr/video-demo",
        filename="eating_spaghetti.mp4",
        repo_type="dataset",
    )
    container = av.open(file_path)

    # sample 32 frames
    indices = sample_frame_indices(
        clip_len=32, frame_sample_rate=4, seg_len=container.streams.video[0].frames
    )
    video = read_video_pyav(container=container, indices=indices)

    from PIL import Image

    videos = []
    video_labels = []
    for x in range(100, len(image_files) - 10, 7):
        video_arr = None
        for y in range(10):
            img = Image.open(f"{image_files[x+y]}")
            img = img.resize((224, 224))
            img_arr = np.array(img)
            if video_arr is None:
                video_arr = np.zeros((10, img_arr.shape[0], img_arr.shape[1], 3))
            video_arr[y, :, :, :] = img_arr
        videos.append(video_arr)
        video_labels.append(labels[x + 9])

    config = VivitConfig.from_pretrained("google/vivit-b-16x2-kinetics400")
    config.num_classes = len(label_names)
    config.id2label = {str(i): c for i, c in enumerate(label_names)}
    config.label2id = {c: str(i) for i, c in enumerate(label_names)}
    config.num_frames = 10
    config.video_size = [10, 224, 224]
    assert videos[0].shape == (
        10,
        224,
        224,
        3,
    ), f"Expected video shape (10, 224, 224, 3), got {videos[0].shape}"

    image_processor = VivitImageProcessor.from_pretrained(
        "google/vivit-b-16x2-kinetics400", do_resize=False, do_center_crop=False
    )
    model = VivitForVideoClassification.from_pretrained(
        "google/vivit-b-16x2-kinetics400",
        config=config,
        ignore_mismatched_sizes=True,
    )

    # inputs = image_processor(list(video), return_tensors="pt")

    # with torch.no_grad():
    #     outputs = model(**inputs)
    #     logits = outputs.logits

    # # model predicts one of the 400 Kinetics-400 classes
    # predicted_label = logits.argmax(-1).item()
    # print(model.config.id2label[predicted_label])

    from transformers import AdamW, Trainer, TrainingArguments

    training_args = TrainingArguments(
        output_dir="./results",
        num_train_epochs=100,
        per_device_train_batch_size=2,
        per_device_eval_batch_size=2,
        learning_rate=1e-03,
        weight_decay=0.01,
        logging_dir="./logs",
        logging_steps=10,
        seed=42,
        evaluation_strategy="steps",
        eval_steps=100,
        warmup_steps=int(0.1 * 20),
        optim="adamw_torch",
        lr_scheduler_type="linear",
        # fp16=True,
        fp16=False,
        load_best_model_at_end=True,
    )

    def process_example(example):
        inputs = image_processor(list(np.array(example["video"])), return_tensors="pt")
        inputs["labels"] = example["labels"]
        return inputs

    from datasets import Dataset

    video_dict = []
    for i, video in enumerate(videos):
        video_dict.append({"video": video, "labels": video_labels[i]})
    dataset = Dataset.from_list(video_dict)
    dataset = dataset.class_encode_column("labels")
    processed_dataset = dataset.map(process_example, batched=False)
    processed_dataset = processed_dataset.remove_columns(["video"])
    shuffled_dataset = processed_dataset.shuffle(seed=42)
    shuffled_dataset = shuffled_dataset.map(
        lambda x: {"pixel_values": torch.tensor(x["pixel_values"]).squeeze()},
    )
    shuffled_dataset = shuffled_dataset.train_test_split(test_size=0.1)

    optimizer = AdamW(model.parameters(), lr=5e-05, betas=(0.9, 0.999), eps=1e-08)
    # Define the trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=shuffled_dataset["train"],
        eval_dataset=shuffled_dataset["test"],
        optimizers=(optimizer, None),
        callbacks=[
            TFEarlyStoppingCallback(3),
            # GradientAccumulationCallback(accumulation_steps=2),
        ],
    )

    train_results = trainer.train()
    trainer.save_model()
    trainer.log_metrics("train", train_results.metrics)
    trainer.save_metrics("train", train_results.metrics)
    trainer.save_state()


if __name__ == "__main__":
    main()
