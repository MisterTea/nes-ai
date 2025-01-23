import copy
import io
import shelve
from pathlib import Path

from PIL import Image

SELECT = 2
START = 3


class RolloutData:
    def __init__(self, data_path: Path):
        self.data_path = data_path
        self.input_images = shelve.open(data_path / "input_images.shelve")
        self.expert_controller = shelve.open(data_path / "expert_controller.shelve")
        self.reward_map_history = shelve.open(data_path / "reward_map.shelve")
        self.reward_vector_history = shelve.open(data_path / "reward_vector.shelve")
        self.agent_params = shelve.open(data_path / "agent_params.shelve")

    def expert_controller_no_start_select(self, frame):
        controller_array = copy.deepcopy(self.expert_controller[frame])
        controller_array[START] = 0
        controller_array[SELECT] = 0
        return controller_array

    def put_image(self, image, frame):
        img_byte_arr = io.BytesIO()
        image.save(img_byte_arr, format="PNG")
        self.input_images[str(frame)] = img_byte_arr.getvalue()

    def get_image(self, frame):
        img_byte_arr = io.BytesIO(self.input_images[str(frame)])
        return Image.open(img_byte_arr)

    def sync(self):
        self.input_images.sync()
        self.expert_controller.sync()
        self.reward_map_history.sync()
        self.reward_vector_history.sync()
        self.agent_params.sync()

    def close(self):
        self.input_images.close()
        self.expert_controller.close()
        self.reward_map_history.close()
        self.reward_vector_history.close()
        self.agent_params.close()
