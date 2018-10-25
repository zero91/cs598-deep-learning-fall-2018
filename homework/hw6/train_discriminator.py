import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms

from discriminator import Discriminator
from data_tools import data_loader_and_transformer

import argparse
def str2bool(v):
    return v.lower() in ("yes", "true", "t", "1")

parser = argparse.ArgumentParser(description="Train the Discriminator without the Generator")

# Hyperparameters.
parser.add_argument("--lr", default=0.0001, type=float, 
                    help="learning rate")
parser.add_argument("--epochs", default=100, type=int, 
                    help="number of training epochs")
parser.add_argument("--batch_size", default=128, type=int, 
                    help="batch size")

# Model options.
parser.add_argument("--load_checkpoint", default=False, type=str2bool, 
                    help="resume from checkpoint")

args = parser.parse_args()


# Load data.
print("==> Loading data...")
trainloader, testloader = data_loader_and_transformer(args.batch_size)

device = 'cuda' if torch.cuda.is_available() else 'cpu'

# Load model.
print("==> Initializing model...")
model = Discriminator()
if args.load_checkpoint:
    print("==> Loading checkpoint...")
    model = torch.load('cifar10.model')
model = model.to(device)

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)


# --------------------------
# Single train and test step
# --------------------------

def train(epoch):
    # Set to train mode.
    model.train()

    # Avoid the potential overflow error from Adam.
    if epoch > 10:
        for group in optimizer.param_groups:
            for p in group['params']:
                state = optimizer.state[p]
                if ('step' in state) and (state['step'] >= 1024):
                    state['step'] = 1000
    
    # Learning rate schedule.
    if epoch == 50:
        for param_group in optimizer.param_groups:
            param_group['lr'] = args.lr / 10.0
    if epoch == 75:
        for param_group in optimizer.param_groups:
            param_group['lr'] = args.lr / 100.0
    
    # To monitor the training process.
    running_loss = 0
    total_correct = 0
    total_samples = 0
    
    # Start training this epoch.
    for batch_index, (images, labels) in enumerate(trainloader):

        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        _, outputs = model(images)

        loss = criterion(outputs, labels)
        loss.backward()

        optimizer.step()

        # Loss.
        running_loss += loss.item()
        curt_loss = running_loss / (batch_index + 1)

        # Accuracy.
        _, predict_label = torch.max(outputs, 1)
        total_samples += labels.shape[0]
        total_correct += predict_label.eq(labels.long()).float().sum().item()
        accuracy = total_correct / total_samples

    print('Training [epoch: %d] loss: %.3f, accuracy: %.5f' %
            (epoch + 1, curt_loss, accuracy))
    
    if epoch + 1 == args.epochs // 5:
        print("=> Saving model checkpoint...")
        torch.save(model,'cifar10.model')

def test(epoch):
    # Set to test mode.
    model.eval()

    # To monitor the testing process.
    running_loss = 0
    total_correct = 0
    total_samples = 0

    # Testing step.
    with torch.no_grad():
        for batch_index, (images, labels) in enumerate(testloader):
            
            images = images.to(device)
            labels = labels.to(device)

            _, outputs = model(images)
            loss = criterion(outputs, labels)
            
            # Loss.
            running_loss += loss.item()
            curt_loss = running_loss / (batch_index + 1)

            # Accuracy
            _, predict_label = torch.max(outputs, 1)
            total_samples += labels.shape[0]
            total_correct += predict_label.eq(labels.long()).float().sum().item()
            accuracy = total_correct / total_samples
    
    print('Testing [epoch: %d] loss: %.3f, accuracy: %.5f' %
            (epoch + 1, curt_loss, accuracy))


# --------------------------
# Start training and testing
# --------------------------

print("==> Start training on device {}...".format(device))
print("\tHyperparameters: LR = {}, EPOCHS = {}, BATCH_SIZE = {}"
        .format(args.lr, args.epochs, args.batch_size))

for epoch in range(args.epochs):
    train(epoch)
    test(epoch)

print("==> Training finished. Saving model checkpoint...")
torch.save(model,'cifar10.model')