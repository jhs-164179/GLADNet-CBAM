import os
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset


def check_path(path):
    if not os.path.exists(path):
        os.makedirs(path)


def load_imgs(path):
    img = Image.open(path).convert('RGB')
    img = np.array(img, dtype='float32') / 255.0
    return img


def save_imgs(path, result1, result2=None):
    result1 = np.squeeze(result1)
    result2 = np.squeeze(result2)

    if not result2.any():
        img = result1
    else:
        img = np.concatenate([result1, result2], axis=1)
    img = Image.fromarray(np.clip(img * 255.0, 0, 255.0).astype('uint8'))
    img.save(path, 'png')


class CustomDataset(Dataset):
    def __init__(self, img_paths, label_paths, transform=True):
        self.img_paths = img_paths
        self.label_paths = label_paths
        self.transform = transform

    def __getitem__(self, index):
        img_path = self.img_paths[index]
        img = load_imgs(img_path)
        if self.transform:
            img = torch.from_numpy(img.transpose((0, 3, 1, 2)))
        else:
            img = torch.from_numpy(img)

        if self.label_paths is not None:
            label_path = self.label_paths[index]
            label = load_imgs(label_path)
            if self.transform:
                label = torch.from_numpy(label.transpose((0, 3, 1, 2)))
            else:
                label = torch.from_numpy(label)

            return img, label
        else:
            return img

    def __len__(self):
        return len(self.img_paths)


# class LowlightEnhance(object):
#     def __init__(self, **kwargs):
#         self.model = kwargs.model

#     def evaluate(self, epoch_num, eval_low_data, sample_dir):
#         print(f'[*] Evaluating for epoch {epoch_num}...')
#         for i in range(len(eval_low_data)):
#             input_low

#     def train(self):
#         pass

#     def save(self):
#         pass

#     def test(self):
#         pass
