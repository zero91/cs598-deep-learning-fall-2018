import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn

import matplotlib.pyplot as plt
import sys
import numpy as np
import random

from data_tools import data_loader_and_transformer
from model import ResNet
from train import train
from test import test
from utils import load_checkpoint
from fine_tune import resnet18

import argparse
def str2bool(v):
    return v.lower() in ("yes", "true", "t", "1")

parser = argparse.ArgumentParser(description="Training ResNet on CIFAR100")
parser.add_argument("--fine_tune", default=False, type=str2bool, help="fine-tune pretrained model")
parser.add_argument("--lr", default=0.001, type=float, help="learning rate")
parser.add_argument("--epochs", default=40, type=int, help="number of training epochs")
parser.add_argument("--lr_schedule", default=True, type=str2bool, help="perform lr shceduling")
parser.add_argument("--load_checkpoint", default=False, type=str2bool, help="resume from checkpoint")
parser.add_argument("--show_sample_image", default=False, type=str2bool, help="display data insights")
parser.add_argument("--debug", default=False, type=str2bool, help="using debug mode")
parser.add_argument("--data_path", default="./data", type=str, help="path to store data")
args = parser.parse_args()


def main():
    """High level pipelines."""

    # Set seed.
    torch.manual_seed(0)
    torch.cuda.manual_seed(0)
    np.random.seed(0)
    random.seed(0)

    # Load data.
    print("*** Performing data augmentation...")
    train_data_loader, test_data_loader = data_loader_and_transformer(
                                                args.data_path, 
                                                fine_tune=args.fine_tune)

    # Load sample image.
    if args.show_sample_image:
        print("*** Loading image sample from a batch...")
        data_iter = iter(train_data_loader)
        images, labels = data_iter.next()  # Retrieve a batch of data
        
        # Some insights of the data.
        # images type torch.FloatTensor, shape torch.Size([128, 3, 32, 32])
        print("images type {}, shape {}".format(images.type(), images.shape))
        # shape of a single image torch.Size([3, 32, 32])
        print("shape of a single image", images[0].shape)
        # labels type torch.LongTensor, shape torch.Size([128])
        print("labels type {}, shape {}".format(labels.type(), labels.shape))
        # label for the first 4 images tensor([12, 51, 91, 36])
        print("label for the first 4 images", labels[:4])
        
        # Get a sampled image.
        plt.imshow(images[0][0].numpy())
        plt.savefig("sample_image.png")

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # Load model.
    if args.fine_tune:
        print("*** Initializing pre-trained model...")
        resnet = resnet18()
        in_features = resnet.fc.in_features
        resnet.fc = nn.Linear(in_features, 100)
    else:
        print("*** Initializing model...")
        resnet = ResNet([2, 4, 4, 2])

    resnet = resnet.to(device)
    if device == 'cuda':
        resnet = torch.nn.DataParallel(resnet)
        cudnn.benchmark = True
    
    # Load checkpoint.
    start_epoch = 0
    best_acc = 0
    if args.load_checkpoint:
        print("*** Loading checkpoint...")
        start_epoch, best_acc = load_checkpoint(resnet)

    # Training.
    print("*** Start training on device {}...".format(device))
    print("* Hyperparameters: LR = {}, EPOCHS = {}, LR_SCHEDULE = {}"
          .format(args.lr, args.epochs, args.lr_schedule))
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(resnet.parameters(), lr=args.lr)
    train(
        resnet,
        criterion,
        optimizer,
        best_acc,
        start_epoch,
        args.epochs,
        train_data_loader,
        test_data_loader,
        device,
        lr_schedule=args.lr_schedule,
        debug=args.debug
    )

    # Testing.
    print("*** Start testing...")
    test(
        resnet,
        criterion,
        test_data_loader,
        device,
        debug=args.debug
    )
    
    print("*** Congratulations! You've got an amazing model now :)")

if __name__=="__main__":
    main()
