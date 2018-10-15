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
    def __init__(self, file_paths, train=True, transform=None):
        self.train = train
        self.transform = transform

        self.image_paths = []
        self.labels = []
        with open(file_paths, "r") as file:
            for line in file:
                path, label = line.strip().split()
                self.image_paths.append(path)
                self.labels.append(label)
   
    def __getitem__(self, index):
        """
        To make it easier to understand, just do online triplet sampling.
        In your getitem(index) function, you should sample a positive and 
        a negative image corresponding the query image reference by the index parameter
        => so don't need the triplet_list.txt file
        """

        if not self.train:
            img = Image.open(self.image_paths[index])
            img = img.convert('RGB')
            label = self.labels[index]
            if self.transform is not None:
                img = self.transform(img)
            return img, label
        
        else:
            step = 500
            count = index // step
            start = count * step
            end = start + step - 1
            total = 100000

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
            q_img = Image.open(self.image_paths[query_idx])
            p_img = Image.open(self.image_paths[positive_idx])
            n_img = Image.open(self.image_paths[negative_idx])
            q_img = q_img.convert('RGB')
            p_img = p_img.convert('RGB')
            n_img = n_img.convert('RGB')

            q_label = self.labels[query_idx]
            p_label = self.labels[positive_idx]
            n_label = self.labels[negative_idx]

            # Transform
            if self.transform is not None:
                q_img = self.transform(q_img)
                p_img = self.transform(p_img)
                n_img = self.transform(n_img)

            # Return tuple of the three images and tuple of the labels
            return (q_img, p_img, n_img), (q_label, p_label, n_label)
    
    def __len__(self):
        return len(self.image_paths)
