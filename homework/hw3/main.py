import torch
import torch.backends.cudnn as cudnn

import matplotlib.pyplot as plt
import sys
import numpy as np
import random

from data_tools import data_loader_and_transformer
from model import DeepCNN
from train import train
from test import test
from utils import load_checkpoint

# Set to True if you have checkpoints available and want to resume from it
LOAD_CHECKPOINT = False

# Set to True to get some insights of the data
SHOW_SAMPLE_IMAGE = False

# Set to True to run in a debug mode which uses less data
DEBUG = False

DATA_PATH = "./data"

# Hyperparameters.
trials = [
    [0.01, 50],
    [0.001, 50],
    [0.01, 100],
    [0.001, 100],
    [0.001, 70],
    [0.001, 15]]

if (len(sys.argv) == 2):
    trial_number = int(sys.argv[1])
else:
    trial_number = 5
LR = trials[trial_number][0]
EPOCHS = trials[trial_number][1]


def main():
    """High level pipelines.
    Usage: run "python3 main.py trial_num"
    such as "python3 main.py 1"
    """

    # Set seed.
    torch.manual_seed(0)
    torch.cuda.manual_seed(0)
    np.random.seed(0)
    random.seed(0)

    # Load data.
    print("*** Performing data augmentation...")
    train_data_loader, test_data_loader = data_loader_and_transformer(DATA_PATH)

    # Load sample image.
    if SHOW_SAMPLE_IMAGE:
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
        # label for the first 4 images tensor([2, 3, 4, 2])
        print("label for the first 4 images", labels[:4])
        
        # Get a sampled image.
        plt.imshow(images[0][0].numpy())
        plt.savefig("sample_image.png")

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # Load model.
    print("*** Initializing model...")
    cnn = DeepCNN()
    # print(cnn)
    cnn = cnn.to(device)
    if device == 'cuda':
        cnn = torch.nn.DataParallel(cnn)
        cudnn.benchmark = True
    
    # Load checkpoint.
    start_epoch = 0
    best_acc = 0
    if LOAD_CHECKPOINT:
        print("*** Loading checkpoint...")
        start_epoch, best_acc = load_checkpoint(cnn)

    # Training.
    print("*** Start training on device {}...".format(device))
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(cnn.parameters(), lr=LR)
    train(
        cnn,
        criterion,
        optimizer,
        best_acc,
        start_epoch,
        EPOCHS,
        train_data_loader,
        device,
        lr_schedule=False,
        debug=DEBUG
    )

    # Testing.
    print("*** Start testing...")
    test(
        cnn,
        criterion,
        test_data_loader,
        device,
        debug=DEBUG
    )
    
    print("*** Congratulations! You've got an amazing model now :)")

if __name__=="__main__":
    main()
