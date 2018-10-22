import torch
import torchvision.transforms as transforms
import torchvision.models as models
import torch.nn as nn
import torch.backends.cudnn as cudnn

from utils import *
from dataset import TinyImageNetDataset
from train import train

import os.path
import multiprocessing
import matplotlib.pyplot as plt
import sys

import argparse
def str2bool(v):
    return v.lower() in ("yes", "true", "t", "1")

parser = argparse.ArgumentParser(description="Image Ranking")

# Hyperparameters.
parser.add_argument("--lr", default=0.001, type=float, 
                    help="learning rate")
parser.add_argument("--epochs", default=20, type=int, 
                    help="number of training epochs")
parser.add_argument("--batch_size", default=16, type=int, 
                    help="batch size")
parser.add_argument("--feature_embedding", default=4096, type=int, 
                    help="dimension of embedded feature")

# Model options.
parser.add_argument("--model", default='resnet50', type=str, 
                    help="name of the chosen ResNet model")

# Training options.
parser.add_argument("--lr_schedule", default=True, type=str2bool, 
                    help="perform lr shceduling")
parser.add_argument("--load_checkpoint", default=False, type=str2bool, 
                    help="resume from checkpoint")
parser.add_argument("--show_sample_image", default=False, type=str2bool, 
                    help="display data insights")
parser.add_argument("--debug", default=False, type=str2bool, 
                    help="using debug mode")

args = parser.parse_args()

models = {
    'resnet18': models.resnet18(pretrained=True), 
    'resnet34': models.resnet34(pretrained=True),
    'resnet50': models.resnet50(pretrained=True),
    'resnet101': models.resnet101(pretrained=True)
}

def main():
    """High level pipelines."""

    # Prepare files for the dataset
    print("==> Peparing files...")

    train_list = "train_list.txt"
    val_list = "val_list.txt"

    if (not os.path.isfile(train_list)) or (not os.path.isfile(val_list)):
        generate_train_img_names("data/tiny-imagenet-200/train/", train_list)
        generate_val_img_names("data/tiny-imagenet-200/val/", val_list)
    
    # Load data.
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

    # num_workers = 32 on BW
    workers = multiprocessing.cpu_count()
    print("\tnumber of workers: {}".format(workers))
    train_loader = torch.utils.data.DataLoader(
        train_set,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=workers
    )

    # Show sample image.
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
    
    # Load model.
    print("==> Loading pretrained {} model...".format(args.model))
    resnet = models[args.model]
    in_features = resnet.fc.in_features
    resnet.fc = nn.Linear(in_features, args.feature_embedding)

    # Use available device.
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    resnet = resnet.to(device)
    if device == 'cuda':
        resnet = torch.nn.DataParallel(resnet)
        cudnn.benchmark = True

    # Load checkpoint.
    start_epoch = 0
    best_loss = sys.maxsize
    chpt_name = 'model_state_' + args.model + '.pt'
    if args.load_checkpoint:
        print("==> Loading checkpoint...")
        start_epoch, best_loss = load_checkpoint(resnet, chpt_name)
    
    # Training.
    print("==> Start training on device {}...".format(device))
    print("\tHyperparameters: LR = {}, EPOCHS = {}, LR_SCHEDULE = {}"
          .format(args.lr, args.epochs, args.lr_schedule))

    criterion = nn.TripletMarginLoss()
    optimizer = torch.optim.SGD(resnet.parameters(), lr=0.001, momentum=0.9)
    train(resnet, criterion, optimizer, 
          best_loss, start_epoch, args.epochs, 
          train_loader, device, 
          chpt_name, 
          lr_schedule=args.lr_schedule, debug=args.debug)


if __name__ == "__main__":
    main()