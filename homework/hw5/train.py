import torch
import torch.nn as nn
import torch.optim as optim
import time

from utils import save_checkpoint

def train(net, criterion, optimizer,
          best_loss, start_epoch, epochs,
          train_data_loader, device, 
          chpt_name,
          lr_schedule=False, debug=False):
    """Training setups
    Args:
        net(class.DeepCNN)
        criterion(torch.nn.CrossEntropyLoss)
        optimizer(torch.optim.Adam)
        best_loss(float): best loss if loaded from checkpoints
        start_epoch(int): start epoch from last checkpoint
        epochs(int): total number of epochs
        train_data_loader(iterator)
        device(str): 'cpu' or 'cuda'
        chpt_name(str): name of the checkpoint to be saved
        lr_schedule(bool): whether to perform leanring rate scheduling
        debug(bool): whether to use a debug mode
    """

    loss_file = open('training_loss.txt', 'w')

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
        total_samples = 0

        # Schedule learning rate if specified.
        if lr_schedule:
            scheduler = optim.lr_scheduler.StepLR(optimizer, 10, gamma=0.01)
            scheduler.step()

        # Traning step.
        start_time = time.time()
        for batch_index, (images, _) in enumerate(train_data_loader):

            # Only train on a smaller subset if debug mode is true
            if debug and total_samples >= 10001:
                return
            
            q_batch, p_batch, n_batch = images
            q_batch = q_batch.to(device)
            p_batch = p_batch.to(device)
            n_batch = n_batch.to(device)

            optimizer.zero_grad()

            # Forward.
            q_out = net(q_batch)
            p_out = net(p_batch)
            n_out = net(n_batch)

            # Backward.
            loss = criterion(q_out, p_out, n_out)
            loss.backward()

            optimizer.step()

            # Loss.
            running_loss += loss.item()
            curt_loss = running_loss / (batch_index + 1)
            
            # Print and write loss to file.
            if ((batch_index + 1) % 100 == 0):
                print('Training [epoch: %d, batch: %d] loss: %.3f' %
                        (curt_epoch + 1, batch_index + 1, curt_loss))
                loss_file.write('{} {} {}'.format(curt_epoch+1, batch_index+1, curt_loss))
            
            # Update best loss and save checkpoint
            if curt_loss < best_loss:
                best_loss = curt_loss
                save_checkpoint(net, curt_epoch, best_loss, chpt_name)

            total_samples += q_batch.shape[0]
        
        # Time for 1 epoch
        print("--- %s seconds for 1 epoch ---" % (time.time() - start_time))
            
    print("Training [finished]")
    loss_file.close()
