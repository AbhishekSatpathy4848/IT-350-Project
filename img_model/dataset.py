import torch
from copy import deepcopy
import random

class StegDataset(torch.utils.data.Dataset):
    def __init__(self, images, transform=None):
        self.images = deepcopy(images)
        self.transform = transform

    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        secret_idx = idx
        while secret_idx == idx:
            secret_idx = random.randint(0, len(self.images) - 1)
        cover = self.transform(self.images[idx])
        secret = self.transform(self.images[secret_idx])
        return torch.cat((cover, secret), 0)