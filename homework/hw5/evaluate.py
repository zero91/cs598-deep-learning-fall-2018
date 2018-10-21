import torch
import torchvision.models as models
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torchvision.transforms as transforms

from utils import load_checkpoint
from dataset import TinyImageNetDataset

import multiprocessing
import numpy as np
from sklearn.neighbors import KDTree, NearestNeighbors
import time

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
    def __init__(self, trained_net, train_set, val_set, train_loader, val_loader, device):
        """Handler for evaluating the training and test accuracy
        Args:
            trained_net: a trained deep ranking model loaded from checkpoint
            train_loader: loader for train data
            val_loader: loader for validation data
        """
        self.net = trained_net
        self.device = device

        # Dataset.
        self.train_set = train_set
        self.val_set = val_set
        self.train_loader = train_loader
        self.val_loader = val_loader

        # Label is a list being converted to 1d no array (num_images,).
        self.train_labels = np.array(self.train_set.get_labels())
        self.val_labels = np.array(self.val_set.get_labels())

        # Get the list of the paths of all images.
        self.train_paths = self.train_set.get_paths()
        self.val_paths = self.val_set.get_paths()

        # For embedding and qury.
        self.embeddings = self._get_embeddings()
        # self.tree = KDTree(self.embeddings)
        self.knn = NearestNeighbors(n_neighbors=30, leaf_size=30, n_jobs=16)
        self.knn.fit(self.embeddings)

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
        
        self.net.eval()

        total_acc = 0
        avg_acc = 0

        start_time = time.time()

        with torch.no_grad():
            for batch_index, (q_images, q_labels) in enumerate(self.train_loader):
                # Shape (batch_size, 3, 224, 224)
                q_images = q_images.to(self.device)
                q_labels = np.array(q_labels)         # Convert tuple to np array (batch_size,)
                q_labels = q_labels.reshape(-1, 1)    # (batch_size, 1)
                batch_size = q_labels.shape[0]

                # Forward pass, shape (batch_size, 4096)
                q_embedding = self.net(q_images).cpu().numpy()

                # Find top 30, shape (batch_size, 30).
                indices = self.knn.kneighbors(q_embedding, n_neighbors=30, return_distance=False)

                # Compute accuracy.
                retrieved_labels = self.train_labels[indices]  # (batch_size, 30)
                
                correct_labels = np.where(retrieved_labels==q_labels)[0]
                correct_count = correct_labels.shape[0]

                curt_acc = correct_count / (30 * batch_size)       # Batch average accuracy
                total_acc += curt_acc
                avg_acc = total_acc / (batch_index + 1)

                if ((batch_index + 1) % 100 == 0):
                    print('[batch: %d] accuracy: %.5f' % (batch_index + 1, avg_acc))
                
                if (batch_index + 1 == 100):
                    print("--- %s seconds for 100 batch ---" % (time.time() - start_time))

        print('[Finished] accuracy: %.5f' % (avg_acc))   
            
        return avg_acc

    def _get_embeddings(self):
        """Get embeddings for all training images"""

        print("Computing feature embeddings...")
        
        self.net.eval()
        embeddings = []

        start_time = time.time()
        
        with torch.no_grad():
            for _, (images, _) in enumerate(self.train_loader):
                # images [batch size, 3, 224, 224]
                images = images.to(self.device)
                # [batch size, 4096]
                batch_embeddings = self.net(images).cpu().numpy()

                # put embed into list
                embeddings += batch_embeddings.tolist()
        
        print("--- %s seconds for getting embeddings ---" % (time.time() - start_time))

        # Shape [num_train_images, embedding_size]
        return np.array(embeddings)
    
    def query(self):
        """Sample 5 images in val set and query the ranking results"""

        # Sample 5 images from different class in val set
        uniques = set()
        while len(uniques) < 5:
            q_indices = np.random.choice(len(self.val_labels), size=5)
            q_labels = self.val_labels[q_indices]
            uniques = set(q_labels)
        
        q_paths = [self.val_paths[idx] for idx in q_indices]

        # Retrieve ranking result
        self.net.eval()
        with torch.no_grad():
            for ite_count, index in enumerate(q_indices):
                q_image, q_label = self.val_set[index]

                print("=> [query {}] label: {}".format(ite_count + 1, q_label))

                # Get feature embedding for query image
                q_image = q_image.to(self.device)
                q_image = q_image.unsqueeze(0)

                # Forward pass.
                q_embedding = self.net(q_image).cpu().numpy()

                # Find top 10 and bottom 10
                total_imgs = len(self.train_set)
                q_embedding = q_embedding.reshape(1, -1)
                # distances, indices = self.tree.query(q_embedding, k=total_imgs)  #[1, total_imgs]
                distances, indices = self.knn.kneighbors(q_embedding, n_neighbors=total_imgs, 
                                                        return_distance=True)

                top10_dist = distances[0][:10]
                top10_idx = indices[0][:10]
                bottom10_dist = distances[0][-10:]
                bottom10_idx = indices[0][-10:]

                print("Top 10 distances and indices:\n{}\n{}"
                    .format(top10_dist, self.train_labels[top10_idx]))
                print("Bottom 10 distances and indices:\n{}\n{}"
                    .format(bottom10_dist, self.train_labels[bottom10_idx]))

                # Write paths, labels and distances to file (format: path, label, dist)
                with open('query_results_' + str(ite_count+1) + '.txt', 'w') as file:
                    # Write query image
                    file.write("{} {} 0\n".format(q_paths[ite_count], q_labels[ite_count]))

                    # Write top 10
                    for i in range(10):
                        idx = top10_idx[i]
                        dist = top10_dist[i]
                        file.write("{} {} {}\n"
                            .format(self.train_paths[idx], self.train_labels[idx], dist))
                    
                    # Write bottom 10
                    for i in range(10):
                        idx = bottom10_idx[i]
                        dist = bottom10_dist[i]
                        file.write("{} {} {}\n"
                            .format(self.train_paths[idx], self.train_labels[idx], dist))


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
    print("Current model is at epoch {} with loss {}".format(start_epoch, best_loss))

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
    print("==> Loading evaluation handler...")
    evaluation = ResultEvaluationHandler(resnet, train_set, val_set,
                                         train_loader, val_loader, device)
    if args.evaluate:
        print("==> Calculating training accuracy...")
        train_acc = evaluation.get_train_accuracy()
        print('Train at [epoch: %d] loss: %.3f, accuracy: %.5f' %
                (start_epoch + 1, best_loss, train_acc))

        print("==> Calculating test accuracy...")
        test_acc = evaluation.get_test_accuracy()
        print('Test at [epoch: %d] accuracy: %.5f' %
                (start_epoch + 1, test_acc))      

    if args.query:
        print("==> Start query...")
        evaluation.query()

if __name__ == '__main__':
    model_analyze()
