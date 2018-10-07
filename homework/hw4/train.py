import torch
import torch.nn as nn
import torch.optim as optim

from test import test

from utils import save_checkpoint

def train(
    net, 
    criterion, 
    optimizer,
    best_acc, 
    start_epoch, 
    epochs,
    train_data_loader,
    test_data_loader,
    device,
    lr_schedule=False,
    debug=False
    ):
    """Training setup for a single epoch
    Args:
        net(class.DeepCNN)
        criterion(torch.nn.CrossEntropyLoss)
        optimizer(torch.optim.Adam)
        best_acc(float): best accuracy if loaded from checkpoints
        start_epoch(int): start epoch from last checkpoint
        epochs(int): total number of epochs
        train_data_loader(iterator)
        device(str): 'cpu' or 'cuda'
        lr_schedule(bool): whether to perform leanring rate scheduling
        debug(bool): whether to use a debug mode
    """

    for curt_epoch in range(start_epoch, epochs):
        # Set to train mode.
        net.train()

        # Avoid the potential overflow error.
        if curt_epoch > 10:
            for group in optimizer.param_groups:
                for p in group['params']:
                    state = optimizer.state[p]
                    if state['step'] >= 1024:
                        state['step'] = 1000

        # To monitor the training process.
        running_loss = 0
        total_correct = 0
        total_samples = 0

        # Schedule learning rate if specified.
        if lr_schedule:
            scheduler = optim.lr_scheduler.StepLR(optimizer, 15, gamma=0.1)
            scheduler.step()

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

            # Loss.
            running_loss += loss.item()
            curt_loss = running_loss / (batch_index + 1)

            # Accuracy.
            _, predict_label = torch.max(outputs, 1)
            total_samples += labels.shape[0]
            total_correct += predict_label.eq(labels.long()).float().sum().item()
            accuracy = total_correct / total_samples
            
            # print('Training [epoch: %d, batch: %d] loss: %.3f, accuracy: %.5f' %
            #         (curt_epoch + 1, batch_index + 1, curt_loss, accuracy))
            
            # Update best accuracy and save checkpoint
            if accuracy > best_acc:
                best_acc = accuracy
                save_checkpoint(net, curt_epoch, best_acc)
        
        # Loss.
        print('Training [epoch: %d] loss: %.3f, accuracy: %.5f' %
                (curt_epoch + 1, curt_loss, accuracy))
        
        # Test every epoch.
        test(net, criterion, test_data_loader, device, debug=debug)
    
    print("Training [finished]")
