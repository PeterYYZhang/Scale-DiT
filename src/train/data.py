import torch
from PIL import Image, ImageFilter, ImageDraw
import cv2
import numpy as np
import os
from torch.utils.data import Dataset
import torchvision.transforms as T
import random
import pandas as pd

class T2IHQDataset(Dataset):
    def __init__(self, 
        base_dataset = None,
        image_height: int = 4096,
        image_width: int = 4096,
        padding: int = 0,
        brute_force_resize: bool = False,
        return_pil_image: bool = False,
        max_tries: int = 10,
        csv_path: str = None,
        ):
        if csv_path is not None:
            self.base_dataset = pd.read_csv(csv_path)
            self.base_dataset = self.base_dataset.to_dict(orient="records")
        else:
            self.base_dataset = base_dataset
        self.image_height = image_height
        self.image_width = image_width
        self.padding = padding
        self.brute_force_resize = brute_force_resize
        self.return_pil_image = return_pil_image
        self.max_tries = max_tries
        assert self.image_width % 1024 == 0 and self.image_height % 1024 == 0, "Image width and height must be multiples of 1024"

    def __len__(self):
        if isinstance(self.base_dataset, pd.DataFrame):
            return len(self.base_dataset)
        else:
            return len(self.base_dataset)
    
    def _resize_image(self, image):
        if self.brute_force_resize:
            image = image.resize((self.image_width, self.image_height), Image.Resampling.LANCZOS)
        else:
            # Resize to the closest resolution (shape must be multiple of 1024) but maintain the original aspect ratio
            orig_w, orig_h = image.size

            # Find the closest multiples of 1024 for width and height
            def closest_multiple(val, base=1024):
                return int(round(val / base) * base)

            # Compute scale to get as close as possible to target, but keep aspect ratio
            # First, scale so that the largest side is as close as possible to the target multiple of 1024
            scale_w = self.image_width / orig_w
            scale_h = self.image_height / orig_h
            scale = min(scale_w, scale_h)

            new_w = int(round(orig_w * scale))
            new_h = int(round(orig_h * scale))

            # Now, round to the closest multiple of 1024
            new_w = max(1024, closest_multiple(new_w, 1024))
            new_h = max(1024, closest_multiple(new_h, 1024))

            # Resize with aspect ratio preserved to fit within (new_w, new_h)
            scale = min(new_w / orig_w, new_h / orig_h)
            resized_w = int(round(orig_w * scale))
            resized_h = int(round(orig_h * scale))
            image = image.resize((resized_w, resized_h), Image.Resampling.LANCZOS)

            # Pad to (new_w, new_h) if needed, centered
            if resized_w != new_w or resized_h != new_h:
                new_image = Image.new("RGB", (new_w, new_h), (0, 0, 0))
                left = (new_w - resized_w) // 2
                top = (new_h - resized_h) // 2
                new_image.paste(image, (left, top))
                image = new_image
            return image
    
    def get_item(self, idx):
        if isinstance(self.base_dataset, pd.DataFrame):
            item = self.base_dataset.iloc[idx]
        else:
            item = self.base_dataset[idx]
        text = item.get('fine_caption', item.get('caption', None))
        if text is not None:
            if "sorry" in text.lower():
                raise ValueError(f"Item {idx} has a sorry caption")
        image = item.get('image_path', None)
        if text is None or image is None:
            raise ValueError(f"Item {idx} has no text or image")
        image = Image.open(image)
        image = self._resize_image(image)
        return {
            'image': image,
            'description': text,
            **({"pil_image": [image]} if self.return_pil_image else {})
        }
    def __getitem__(self, index):
        for _ in range(self.max_tries):
            try:
                return self.get_item(index)
            except Exception as e:
                print(f"Error getting item {index}: {e}")
                index = random.randint(0, len(self.base_dataset) - 1)
        raise Exception(f"Failed to get item {index} after {self.max_tries} tries")
    

class ImageConditionDataset(Dataset):
    def __init__(
        self,
        base_dataset,
        target_size: int = 512,
        drop_text_prob: float = 0.1,
        return_pil_image: bool = False,
        max_tries: int = 10,
    ):
        self.base_dataset = base_dataset
        self.target_size = target_size
        self.drop_text_prob = drop_text_prob
        self.return_pil_image = return_pil_image
        self.max_tries = max_tries

        self.to_tensor = T.ToTensor()

    def __len__(self):
        return len(self.base_dataset)

    def get_item(self, idx):
        # The base dataset is expected to have "jpg" and "json" columns.
        item = self.base_dataset[idx]
        
        # # Validate that required keys exist and are not None
        # if "image" not in item or item["image"] is None:
        #     raise ValueError(f"Item {idx} has no image or image is None")
        # if "prompt" not in item or item["prompt"] is None:
        #     raise ValueError(f"Item {idx} has no prompt or prompt is None")
        try:
            image = item["image"]
            text = item["prompt"]
        except:
            image = item["jpg"]
            text = item["json"]["prompt"]

        # If the image has an alpha channel, convert it to RGB
        if image.mode == "RGBA":
            image = image.convert("RGB")

        image = image.resize((self.target_size, self.target_size)).convert("RGB")
        description = text

        # Randomly drop text or image
        if random.random() < self.drop_text_prob:
            description = ""

        return {
            "image": self.to_tensor(image),
            "description": description,
            **({"pil_image": [image]} if self.return_pil_image else {}),
        }

    def __getitem__(self, idx):
        for _ in range(self.max_tries):
            try:
                return self.get_item(idx)
            except Exception as e:
                print(f"Error getting item {idx}: {e}")
                idx = random.randint(0, len(self.base_dataset) - 1)
        raise Exception(f"Failed to get item {idx} after {self.max_tries} tries")