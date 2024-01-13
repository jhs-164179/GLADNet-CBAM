import os
from PIL import Image
import torch
from torch.utils.data import Dataset


def check_path(path):
    if not os.path.exists(path):
        os.makedirs(path)


def load_imgs(path):
    Image.open(path)


def save_imgs(path):
    pass


class CustomDataset(Dataset):
    def __init__(self, img_paths, label_paths, transform=None):
        self.img_paths = img_paths
        self.label_paths = label_paths
        self.transform = transform

    def __getitem(self, index):
        img_path = self.img_paths[index]

        img = Image.open(img_path).convert('RGB')
        if self.transform is not None:
            img = self.transform(img)

        if self.label_paths is not None:
            label = torch.tensor(self.label_paths[index], dtype=torch.float) - 1
            return img, label
        else:
            return img

    def __len__(self):
        return len(self.img_paths)


class LowlightEnhance(object):
    def __init__(self, **kwargs):
        self.model = kwargs.model

    def evaluate(self):
        pass

    def train(self):
        pass

    def save(self):
        pass

    def test(self):
        pass