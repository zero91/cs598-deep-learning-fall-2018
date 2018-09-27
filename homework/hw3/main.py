import torch
import torch.optim as optim
import torch.backends.cudnn as cudnn

from data_tools import data_loader_and_transformer
from model import DeepCNN
from train import train_single_epoch
from test import test_single_epoch
from utils import load_checkpoint

import matplotlib.pyplot as plt

# Configurations.
LOAD_CHECKPOINT = False
SHOW_SAMPLE_IMAGE = False
DEBUG = True
DATA_PATH = "./data"

# Hyperparameters.
LR = 0.001
EPOCHS = 50

def main():
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

    # Train and validate.
    print("*** Start training on device {}...".format(device))
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = optim.Adam(cnn.parameters(), lr=LR)

    for epoch in range(start_epoch, EPOCHS):
        train_single_epoch(
            cnn,
            criterion,
            optimizer,
            epoch,
            train_data_loader,
            device,
            lr_schedule=False,
            debug=True
        )
        best_acc = test_single_epoch(
            cnn,
            criterion,
            best_acc,
            epoch,
            test_data_loader,
            device,
            debug=True
        )
    
    print("*** Congratulations! You've got an amazing model now :)")

if __name__=="__main__":
    main()
