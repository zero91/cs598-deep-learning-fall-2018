from os import listdir
from os.path import join
import random

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