import torch
import torch.nn as nn
import torch.optim as optim

def train_single_epoch(net, criterion, optimizer, 
                       curt_epoch,
                       train_data_loader, device,
                       lr_schedule=False,
                       debug=False):
    """Training setup for a single epoch
    Args:
        net(class.DeepCNN)
        criterion(torch.nn.CrossEntropyLoss)
        optimizer(torch.optim.Adam)
        curt_epoch(int): current training epoch
        train_data_loader(iterator)
        device(str): 'cpu' or 'cuda'
        lr_schedule(bool): whether to perform leanring rate scheduling
        debug(bool): whether to use a debug mode
    """

    # Set to train mode.
    net.train()

    # To monitor the training process.
    running_loss = 0
    total_correct = 0
    total_samples = 0

    # Schedule learning rate if specified.
    if lr_schedule:
        optim.lr_scheduler.StepLR(optimizer, 30, gamma=0.1)

    # Traning step.
    for batch_index, (images, labels) in enumerate(train_data_loader):
        # Only train on a smaller subset if debug mode is true
        if debug and total_samples >= 10001:
            return
        
        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        outputs = net(images)

        # CrossEntropyLoss wants inputs as torch.FloatTensor, and 
        # targets as torch.LongTensor which are class indices, not float values.
        loss = criterion(outputs, labels.long())
        loss.backward()

        optimizer.step()

        # Training statistics.
        running_loss += loss.item()
        _, predict_label = torch.max(outputs, 1)
        total_samples += labels.shape[0]
        total_correct += predict_label.eq(labels.long()).float().sum().item()
        accuracy = total_correct / total_samples
        print('Training [epoch: %d, batch: %d] loss: %.3f, accuracy: %.5f' %
                (curt_epoch + 1, batch_index + 1, 
                running_loss / (batch_index + 1), accuracy))
