import torch
from PIL import Image
from torchvision import transforms

from nes.ai.helpers import upscale_and_get_labels

DEFAULT_TRANSFORM = transforms.Compose(
    [
        transforms.ToTensor(),
        transforms.Resize((224, 224)),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]
)


class NESDataset(torch.utils.data.Dataset):
    def __init__(self):
        self.image_files_upscaled, self.file_label_map = upscale_and_get_labels()
        self.label_int_map = {}
        for label in self.file_label_map.values():
            if label not in self.label_int_map:
                self.label_int_map[label] = len(self.label_int_map)
        print("LABEL SIZE", len(self.label_int_map))

    def __len__(self):
        return len(self.image_files_upscaled) - 10

    def get_image_at_frame(self, frame):
        image_filename = self.image_files_upscaled[frame]
        image = Image.open(image_filename)
        image = DEFAULT_TRANSFORM(image)
        return image

    def __getitem__(self, idx):
        image_stack = torch.stack([self.get_image_at_frame(idx + i) for i in range(4)])

        return (
            image_stack,
            self.label_int_map[
                self.file_label_map[self.image_files_upscaled[idx].name]
            ],
        )
