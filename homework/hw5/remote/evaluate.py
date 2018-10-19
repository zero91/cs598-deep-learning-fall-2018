import torch
import torchvision.models as models
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torchvision.transforms as transforms

from utils import load_checkpoint
from dataset import TinyImageNetDataset

import multiprocessing
import numpy as np
from sklearn.neighbors import KDTree

import argparse
def str2bool(v):
    return v.lower() in ("yes", "true", "t", "1")

parser = argparse.ArgumentParser(description="Evaluate the trained model")
parser.add_argument("--batch_size", default=30, type=int, 
                    help="batch size for loading images")
parser.add_argument("--model", default='resnet34', type=str, 
                    help="name of the trained ResNet model")

# Evaluation options.
parser.add_argument("--evaluate", default=True, type=str2bool, 
                    help="evaluate training and validation loss")
parser.add_argument("--query", default=False, type=str2bool, 
                    help="perform query on 5 random images")

args = parser.parse_args()


models = {
    'resnet18': models.resnet18(pretrained=True), 
    'resnet34': models.resnet34(pretrained=True),
    'resnet50': models.resnet50(pretrained=True),
    'resnet101': models.resnet101(pretrained=True)
}


class ResultEvaluationHandler:
    def __init__(self, trained_net, train_set, val_set, train_loader, val_loader):
        """Handler for evaluating the training and test accuracy
        Args:
            trained_net: a trained deep ranking model loaded from checkpoint
            train_loader: loader for train data
            val_loader: loader for validation data
        """
        # Trained model.
        self.net = trained_net
        self.net.eval()

        # Dataset.
        self.train_set = train_set
        self.val_set = val_set
        self.train_loader = train_loader
        self.val_loader = val_loader

        # Label is a list being converted to 1d no array (num_images,)
        self.train_labels = np.array(self.train_set.get_labels())
        self.val_labels = np.array(self.val_set.get_labels())

        self.embeddings = self._get_embeddings()
        self.tree = KDTree(self.embeddings)

    def get_train_accuracy(self):
        return self._get_avg_accuracy(True)

    def get_test_accuracy(self):
        return self._get_avg_accuracy(False)
    
    def _get_avg_accuracy(self, get_train_acc):
        """Get average accuracy accross all query images"""

        if get_train_acc:
            dataset = self.train_set
        else:
            dataset = self.val_set
        
        num_images = len(dataset)
        total_acc = 0

        for i in range(num_images):
            q_image, q_label = dataset[i]
            q_embedding = self.net(q_image).numpy()

            # Convert 1d vector to 2d in shape [1, 4096]
            q_embedding = q_embedding.reshape(1, -1)

            # Find top 30, shape [1, 30]
            indices = self.tree.query(q_embedding, k=30, return_distance=False)

            # Compute accuracy
            retrieved_labels = self.train_labels[indices[0]]  # (30,)
            correct_labels = np.where(retrieved_labels==q_label)[0]
            correct_count = correct_labels.shape[0]
            curt_acc = correct_count / 30
            total_acc += curt_acc
        
        avg_acc = total_acc / num_images
        return avg_acc

    def _get_embeddings(self):
        """Get embeddings for all training images"""
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
    
    def query(self, label):
        """Sample an image of a class in val set and query the ranking results"""

def model_analyze():
    print("==> Start analyzing model...")

    print("==> Loading trained {} model from disk...".format(args.model))
    resnet = models[args.model]
    in_features = resnet.fc.in_features
    resnet.fc = nn.Linear(in_features, 4096)

    # Use available device.
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    resnet = resnet.to(device)
    if device == 'cuda':
        resnet = torch.nn.DataParallel(resnet)
        cudnn.benchmark = True

    # Load trained model from disk.
    chpt_name = 'model_state_' + args.model + '.pt'
    start_epoch, best_loss = load_checkpoint(resnet, chpt_name)

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

    # Perform evaluation
    evaluation = ResultEvaluationHandler(resnet, train_set, val_set,
                                         train_loader, val_loader)
    if args.evaluate:
        print("==> Evaluating accuracies...")

        train_acc = evaluation.get_train_accuracy()
        print('Train at [epoch: %d] loss: %.3f, accuracy: %.5f' %
                (start_epoch + 1, best_loss, train_acc))

        test_acc = evaluation.get_test_accuracy()
        print('Test at [epoch: %d] accuracy: %.5f' %
                (start_epoch + 1, test_acc))      

    if args.query:
        pass

# TODO: query function
# TODO: plot loss should be in another file and run separately


if __name__ == '__main__':
    model_analyze()
