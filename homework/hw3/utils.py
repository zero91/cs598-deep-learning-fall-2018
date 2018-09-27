import os
import torch

def load_checkpoint(net):
    print("Loading model from disk...")

    if not os.path.isdir('checkpoints'):
        print("Error: no checkpoints available.")
        raise AssertionError()
    
    checkpoint = torch.load('checkpoints/model_state.pt')
    net.load_state_dict(checkpoint['model_state_dict'])
    start_epoch = checkpoint['epoch']
    best_acc = checkpoint['best_acc']

    return start_epoch, best_acc


def save_checkpoint(net, epoch, best_acc):
    print("Saving model to disk...")

    state = {
        'model_state_dict': net.state_dict(),
        'epoch': epoch,
        'best_acc': best_acc
    }
    if not os.path.isdir('checkpoints'):
        os.mkdir('checkpoints')
    
    torch.save(state, 'model_state.pt')