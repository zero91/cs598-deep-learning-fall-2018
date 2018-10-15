import torch
import torchvision.transforms as transforms

from utils import generate_train_img_names, generate_val_img_names
from dataset import TinyImageNetDataset

import os.path
import multiprocessing
import matplotlib.pyplot as plt

import argparse
def str2bool(v):
    return v.lower() in ("yes", "true", "t", "1")

parser = argparse.ArgumentParser(description="Image Ranking")
parser.add_argument("--lr", default=0.001, type=float, help="learning rate")
parser.add_argument("--epochs", default=40, type=int, help="number of training epochs")
# parser.add_argument("--lr_schedule", default=True, type=str2bool, help="perform lr shceduling")
# parser.add_argument("--load_checkpoint", default=False, type=str2bool, help="resume from checkpoint")
parser.add_argument("--show_sample_image", default=False, type=str2bool, help="display data insights")
# parser.add_argument("--debug", default=False, type=str2bool, help="using debug mode")
# parser.add_argument("--data_path", default="./data", type=str, help="path to store data")
args = parser.parse_args()

def main():
    """High level pipelines."""

    # Prepare files for the dataset
    print("==> Peparing files...")

    train_list = "train_list.txt"
    val_list = "val_list.txt"

    if (not os.path.isfile(train_list)) or (not os.path.isfile(val_list)):
        generate_train_img_names("data/tiny-imagenet-200/train/", train_list)
        generate_val_img_names("data/tiny-imagenet-200/val/", val_list)
    
    # Load data
    print("==> Loading data...")
    train_set = TinyImageNetDataset(
        train_list, 
        train=True,
        transform=transforms.Compose([
            transforms.Resize(224),
            transforms.RandomCrop(224, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
        ])
    )
    val_set = TinyImageNetDataset(
        val_list,
        train=False,
        transform=transforms.Compose([
            transforms.Resize(224),
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        ])
    )

    # num_workers = 32 on BW
    workers = multiprocessing.cpu_count()
    print("\tnumber of workers: {}".format(workers))
    train_loader = torch.utils.data.DataLoader(
        train_set,
        batch_size=128,
        shuffle=True,
        num_workers=workers
    )
    val_loader = torch.utils.data.DataLoader(
        val_set,
        batch_size=128,
        shuffle=False,
        num_workers=workers
    )

    # Show sample image
    if args.show_sample_image:
        print("==> Loading image sample from a batch...")
        data_iter = iter(train_loader)
        images, labels = data_iter.next()  # Retrieve a batch of data
        
        # All in shape torch.Size([128, 3, 224, 224])
        print("\tlist (len = {}) of batch images".format(len(images)))
        print("\ta batch of query images in shape {}".format(images[0].shape))
        print("\ta batch of positive images in shape {}".format(images[1].shape))
        print("\ta batch of negative images in shape {}".format(images[2].shape))
        print("\tshape of labels within a batch {}".format(len(labels[0])))
        
        # Get a sampled triplet.
        fig = plt.figure()
        titles = ["query", "positive", "negative"]
        for i in range(len(images)):
            sample = images[i][0][0].numpy()
            ax = plt.subplot(1, 3, i + 1)
            plt.tight_layout()
            ax.set_title(titles[i])
            ax.axis('off')
            plt.imshow(sample)
            plt.savefig("sampled_triplet.png")

            if i == 2:
                plt.show()
                break
    

    # TODO: Load ResNet101, write utils for checkpoint, train and test


if __name__ == "__main__":
    main()