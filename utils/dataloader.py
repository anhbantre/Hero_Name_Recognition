import torch
from PIL import Image
import albumentations as A
from torchvision import datasets
from torch.utils.data import Dataset, DataLoader
from albumentations.pytorch.transforms import ToTensorV2

import numpy as np

from utils.utils import label_img


class HeroNameDataset(Dataset):
    """
    The dataloader for loading data to train
    Args:
        data_path (str) -- A path of folder leads to all image paths
        split_data (list) -- A list stores the names of images
        label_path (str) -- A path of a file that contains all names of heros
    """
    def __init__(self, data_path, split_data, label_path):
        self.data_path = data_path
        self.label_path = label_path
        self.list_img_name = split_data
        self.transform = A.Compose([
            A.Resize(height=50, width=100, always_apply=True),
            A.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            ToTensorV2(always_apply=True)
        ])

        # Label images
        self.labeled_img = label_img(self.list_img_name, self.label_path)

    def __len__(self):
        return int(len(self.list_img_name))

    def __getitem__(self, idx):
        # Load data
        img = Image.open(f'{self.data_path}/{self.list_img_name[idx]}').convert('RGB')
        w, h = img.size

        # Cut the image into two equal parts and take the left part
        img = img.crop((0, 0, w/2, h))
        img = np.array(img)
        img = self.transform(image = img)["image"]

        # Get the label
        label = self.labeled_img[self.list_img_name[idx]]

        return img, label
