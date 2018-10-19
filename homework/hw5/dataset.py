from PIL import Image
from torch.utils.data import Dataset

import random

class TinyImageNetDataset(Dataset):
    def __init__(self, file_paths, train=True, transform=None):
        self.train = train
        self.transform = transform

        # Load image paths
        self.image_paths = []
        self.labels = []
        with open(file_paths, "r") as file:
            for line in file:
                path, label = line.strip().split()
                self.image_paths.append(path)
                self.labels.append(label)
        
        # Load all images
        self.images = []
        for path in self.image_paths:
            img = Image.open(path)
            img = img.convert('RGB')
            if self.transform is not None:
                img = self.transform(img)
            self.images.append(img)
        
        print("loaded {} images, each in shape {}".format(len(self.images), self.images[0].numpy().shape))
   
    def get_all_images(self):
        return self.images
    
    def get_all_labels(self):
        return self.labels

    def __getitem__(self, index):
        """Sample triplets online."""

        if not self.train:
            img = self.images[index]
            label = self.labels[index]
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
            q_img = self.images[query_idx]
            p_img = self.images[positive_idx]
            n_img = self.images[negative_idx]

            q_label = self.labels[query_idx]
            p_label = self.labels[positive_idx]
            n_label = self.labels[negative_idx]

            # Return tuple of the three images
            return (q_img, p_img, n_img), (q_label, p_label, n_label)
    
    def __len__(self):
        return len(self.image_paths)