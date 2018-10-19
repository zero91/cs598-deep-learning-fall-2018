import torch
import torchvision.models as models
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torchvision.transforms as transforms

from utils import load_checkpoint
from dataset import TinyImageNetDataset

import multiprocessing
import numpy as np

import argparse
def str2bool(v):
    return v.lower() in ("yes", "true", "t", "1")

parser = argparse.ArgumentParser(description="Evaluate the trained model")
parser.add_argument("--plot_loss", default=False, type=str2bool, help="plot training loss")
parser.add_argument("--query", default=False, type=str2bool, help="perform query on 5 random images")
parser.add_argument("--batch_size", default=10, type=int, help="batch size")

args = parser.parse_args()

class ResultEvaluationHandler:
    def __init__(self, trained_net, train_set, val_set, train_loader, val_loader):
        """Handler for evaluating the training and test accuracy
        Args:
            trained_net: a trained deep ranking model loaded from checkpoint
            train_loader: loader for train data
            val_loader: loader for validation data
        """
        self.net = trained_net
        self.train_set = train_set
        self.val_set = val_set
        self.train_loader = train_loader
        self.val_loader = val_loader

        self.train_labels = self.train_set.get_labels()
        self.val_labels = self.val_set.get_labels()

        self.embeddings = self._get_embeddings()

    def get_train_accuracy(self):
        pass


    def get_test_accuracy(self):
        pass
    
    def _get_avg_accuracy(self, get_train_acc):
        if get_train_acc:
            dataset = self.train_set
        else:
            dataset = self.val_set
        
        num_images = len(dataset)
        accuracy = 0

        for i in range(num_images):
            curt_image, curt_label = dataset[i]
            

    def _get_embeddings(self):
        embeddings = []
        for _, (images, _) in enumerate(self.train_loader):
            # images [batch size, 3, 224, 224]
            # labels: list of length (batch_size)

            # [batch size, 4096]
            batch_embeddings = self.net(images)
            batch_embeddings = batch_embeddings.numpy()

            # put embed into list
            embeddings += batch_embeddings.tolist()
        
        # Shape [num_train_images, embedding_size]
        return np.array(embeddings)
    
    def polt_loss(self, path):
        loss_file = open(path, 'r')


        loss_file.close()
    
    def query(self, label):
        """Sample an image of a class in val set and query the ranking results"""


def evaluate():
    print("==> Loading trained model from disk...")

    # Model.
    resnet = models.resnet101(pretrained=True)
    in_features = resnet.fc.in_features
    resnet.fc = nn.Linear(in_features, 4096)

    # Use available device.
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    resnet = resnet.to(device)
    if device == 'cuda':
        resnet = torch.nn.DataParallel(resnet)
        cudnn.benchmark = True

    # Load trained model from disk.
    _, _ = load_checkpoint(resnet)

    # Load all training and validation images.
    print("==> Loading images...")

    train_list = "train_list.txt"
    val_list = "val_list.txt"
    
    train_set = TinyImageNetDataset(
        train_list,
        train=False,
        transform=transforms.Compose([
            transforms.Resize(224),
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
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

    workers = multiprocessing.cpu_count()
    print("\tnumber of workers: {}".format(workers))
    train_loader = torch.utils.data.DataLoader(
        train_set,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=workers
    )
    val_loader = torch.utils.data.DataLoader(
        val_set,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=workers
    )


def plot():
    pass

def query():
    pass

if __name__ == '__main__':
    evaluate()

    if args.plot_loss:
        plot()
    if args.query:
        query()
    
