import torch
from PIL import Image
import os
import os.path
import numpy as np
from torch.utils.data import Dataset, DataLoader
import torchvision
import torchvision.transforms as transforms

import random

class TinyImageNetDataset(Dataset):
    def __init__(self, root, train=True, transform=None):
        self.train = train
    
    def __getitem__(self, index):
        """
        To make it easier to understand, just do online triplet sampling.
        In your getitem(index) function, you should sample a positive and 
        a negative image corresponding the query image reference by the index parameter
        => so don't need the triplet_list.txt file
        """

        step = 500
        count = index // step
        start = count * step
        end = start + step - 1

        # Query image
        query_idx = index

        # Sample positive image within the same class
        positive_idx = random.randint(start, end)
        while positive_idx == query_idx:
            positive_idx = random.randint(start, end)
        
        # Sample positive image in other class
        range1 = list(range(0, start))
        range2 = list(range(end + 1, total))
        negative_idx = random.sample(range1 + range2, 1)[0]

        # Load the three images

        pass
    
    def __len__(self):
        pass
