import os
from os import listdir
from os.path import join
import random
import torch

def generate_train_img_names(root_path, out_path):
    """Get path and label of training images.
    Produce a txt file where each line contains the path and label separated by sapce,
    such as 'data/tiny-imagenet-200/train/n01443537/images/n01443537_0.JPEG n01443537'
    Args:
        root_path(str): similar to "data/tiny-imagenet-200/train/"
        out_path(str): txt file to store the results
    """

    with open(out_path, "w") as file:
        for class_name in listdir(root_path):
            if class_name[0] == "n":
                full_path = root_path + class_name + "/images"
                for image in listdir(full_path):
                    file.write(join(full_path, image) + " " + class_name + "\n")

def generate_val_img_names(root_path, out_path):
    """Get path and label of validate images.
    Produce a txt file where each line contains the path and label separated by sapce,
    such as 'data/tiny-imagenet-200/val/images/val_0.JPEG n03444034'
    Args:
        root_path(str): similar to "data/tiny-imagenet-200/val/"
        out_path(str): txt file to store the results
    """

    out = open(out_path, "w")
    
    with open(root_path + "val_annotations.txt", "r") as file:
        for line in file:
            image_name, label = line.split()[0], line.split()[1]
            out.write(root_path + "images/" + image_name + " " + label + "\n")
    
    out.close()

def load_checkpoint(net, chpt_name):
    print("Loading model from disk...")

    if not os.path.isdir('checkpoints'):
        print("Error: no checkpoints available.")
        raise AssertionError()
    
    checkpoint = torch.load('checkpoints/' + chpt_name)
    net.load_state_dict(checkpoint['model_state_dict'])
    start_epoch = checkpoint['epoch']
    best_loss = checkpoint['best_loss']

    return start_epoch, best_loss


def save_checkpoint(net, epoch, best_loss, chpt_name):
    """Save checkpoint to disk
    Args:
        net(model class)
        epoch(int): current epoch number
        best_loss(float): best training loss till now
    """

    state = {
        'model_state_dict': net.state_dict(),
        'epoch': epoch,
        'best_loss': best_loss
    }
    if not os.path.isdir('checkpoints'):
        os.mkdir('checkpoints')
    
    torch.save(state, 'checkpoints/' + chpt_name)