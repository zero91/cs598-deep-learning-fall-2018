from utils import generate_train_img_names, generate_val_img_names

import os.path

import argparse
# def str2bool(v):
#     return v.lower() in ("yes", "true", "t", "1")

parser = argparse.ArgumentParser(description="Image Ranking")
parser.add_argument("--lr", default=0.001, type=float, help="learning rate")
parser.add_argument("--epochs", default=40, type=int, help="number of training epochs")
# parser.add_argument("--lr_schedule", default=True, type=str2bool, help="perform lr shceduling")
# parser.add_argument("--load_checkpoint", default=False, type=str2bool, help="resume from checkpoint")
# parser.add_argument("--show_sample_image", default=False, type=str2bool, help="display data insights")
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

if __name__ == "__main__":
    main()