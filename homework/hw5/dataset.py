import torch
from PIL import Image
import os
import os.path
import numpy as np
from torch.utils.data import Dataset, DataLoader
import torchvision
import torchvision.transforms as transforms

class TinyImageNetDataset(Dataset):
    def __init__(self, root, train=True, transform=None):
        self.train = train
    
    def __getitem__(self, index):
        pass
    
    def __len__(self):
        pass
