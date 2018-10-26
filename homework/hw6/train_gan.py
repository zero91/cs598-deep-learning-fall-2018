import torch
import torch.nn as nn
from torch import autograd
from torch.autograd import Variable
import torchvision
import torchvision.transforms as transforms

from discriminator import Discriminator
from generator import Generator
from data_tools import data_loader_and_transformer

import multiprocessing
import numpy as np
import time

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import os

import argparse
def str2bool(v):
    return v.lower() in ("yes", "true", "t", "1")

parser = argparse.ArgumentParser(description="Train the Discriminator without the Generator")

# Hyperparameters.
parser.add_argument("--lr", default=0.0001, type=float, 
                    help="learning rate")
parser.add_argument("--epochs", default=200, type=int, 
                    help="number of training epochs")
parser.add_argument("--batch_size", default=128, type=int, 
                    help="batch size")

# Model options.
parser.add_argument("--load_checkpoint", default=False, type=str2bool, 
                    help="resume from checkpoint")

args = parser.parse_args()
batch_size = args.batch_size
num_epochs = args.epochs


# ----------------
# Helper functions
# ----------------

def calc_gradient_penalty(netD, real_data, fake_data):
    DIM = 32
    LAMBDA = 10
    alpha = torch.rand(batch_size, 1)
    alpha = alpha.expand(batch_size, int(real_data.nelement()/batch_size)).contiguous()
    alpha = alpha.view(batch_size, 3, DIM, DIM)
    alpha = alpha.cuda()
    
    fake_data = fake_data.view(batch_size, 3, DIM, DIM)
    interpolates = alpha * real_data.detach() + ((1 - alpha) * fake_data.detach())

    interpolates = interpolates.cuda()
    interpolates.requires_grad_(True)

    disc_interpolates, _ = netD(interpolates)

    gradients = autograd.grad(outputs=disc_interpolates, inputs=interpolates,
                              grad_outputs=torch.ones(disc_interpolates.size()).cuda(),
                              create_graph=True, retain_graph=True, only_inputs=True)[0]

    gradients = gradients.view(gradients.size(0), -1)                              
    gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean() * LAMBDA
    
    return gradient_penalty

def plot(samples):
    fig = plt.figure(figsize=(10, 10))
    gs = gridspec.GridSpec(10, 10)
    gs.update(wspace=0.02, hspace=0.02)

    for i, sample in enumerate(samples):
        ax = plt.subplot(gs[i])
        plt.axis('off')
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        ax.set_aspect('equal')
        plt.imshow(sample)
    
    return fig


# --------------
# Training setup
# --------------

# Load data.
print("==> Loading data...")
trainloader, testloader = data_loader_and_transformer(args.batch_size)

# Load model.
print("==> Initializing model...")

aD = Discriminator()
aG = Generator()

if args.load_checkpoint:
    print("==> Loading checkpoint...")
    aD = torch.load('tempD.model')
    aG = torch.load('tempG.model')

aD.cuda()
aG.cuda()

# Optimizers, one for each model.
optimizer_g = torch.optim.Adam(aG.parameters(), lr=0.0001, betas=(0, 0.9))
optimizer_d = torch.optim.Adam(aD.parameters(), lr=0.0001, betas=(0, 0.9))

criterion = nn.CrossEntropyLoss()

# Create random batch of noise for the generator.
print("==> Generating noise...")
n_z = 100
n_classes = 10

np.random.seed(352)
label = np.asarray(list(range(10))*10)
noise = np.random.normal(0, 1, (100, n_z))
label_onehot = np.zeros((100, n_classes))
label_onehot[np.arange(100), label] = 1
noise[np.arange(100), :n_classes] = label_onehot[np.arange(100)]
noise = noise.astype(np.float32)

save_noise = torch.from_numpy(noise)
save_noise = Variable(save_noise).cuda()


# --------------------------
# Start training and testing
# --------------------------

start_time = time.time()

# Train the model.
print("==> Start training...")
print("\tHyperparameters: LR = {}, EPOCHS = {}, BATCH_SIZE = {}"
        .format(args.lr, args.epochs, args.batch_size))

gen_train = 1

for epoch in range(0, num_epochs):
    aG.train()
    aD.train()

    # Avoid the potential overflow error from Adam.
    if epoch > 10:
        for group in optimizer_g.param_groups:
            for p in group['params']:
                state = optimizer_g.state[p]
                if ('step' in state) and (state['step'] >= 1024):
                    state['step'] = 1000
        
        for group in optimizer_d.param_groups:
            for p in group['params']:
                state = optimizer_d.state[p]
                if ('step' in state) and (state['step'] >= 1024):
                    state['step'] = 1000

    # To monitor the training process.
    loss1 = []
    loss2 = []
    loss3 = []
    loss4 = []
    loss5 = []
    acc1 = []

    for batch_idx, (X_train_batch, Y_train_batch) in enumerate(trainloader):
        if Y_train_batch.shape[0] < batch_size:
            continue
        
        # Train the generator G.
        if batch_idx % gen_train == 0:

            # Turn the gradient for D off to save GPU memory. 
            for p in aD.parameters():
                p.requires_grad_(False)

            aG.zero_grad()

            # Create random batch of noise for the generator.
            label = np.random.randint(0, n_classes, batch_size)
            noise = np.random.normal(0, 1, (batch_size, n_z))
            label_onehot = np.zeros((batch_size, n_classes))
            label_onehot[np.arange(batch_size), label] = 1
            noise[np.arange(batch_size), :n_classes] = label_onehot[np.arange(batch_size)]
            noise = noise.astype(np.float32)
            noise = torch.from_numpy(noise)
            noise = Variable(noise).cuda()
            fake_label = Variable(torch.from_numpy(label)).cuda()

            fake_data = aG(noise)
            gen_source, gen_class  = aD(fake_data)

            gen_source = gen_source.mean()
            gen_class = criterion(gen_class, fake_label)

            gen_cost = -gen_source + gen_class
            gen_cost.backward()

            optimizer_g.step()
        
        # Train the discriminator D.
        for p in aD.parameters():
            p.requires_grad_(True)

        aD.zero_grad()

        # Train discriminator with input from generator.
        label = np.random.randint(0, n_classes, batch_size)
        noise = np.random.normal(0, 1, (batch_size, n_z))
        label_onehot = np.zeros((batch_size, n_classes))
        label_onehot[np.arange(batch_size), label] = 1
        noise[np.arange(batch_size), :n_classes] = label_onehot[np.arange(batch_size)]
        noise = noise.astype(np.float32)
        noise = torch.from_numpy(noise)
        noise = Variable(noise).cuda()
        fake_label = Variable(torch.from_numpy(label)).cuda()
        with torch.no_grad():
            fake_data = aG(noise)

        disc_fake_source, disc_fake_class = aD(fake_data) 
        disc_fake_source = disc_fake_source.mean()
        disc_fake_class = criterion(disc_fake_class, fake_label)

        # Train discriminator with input from the discriminator.
        real_data = Variable(X_train_batch).cuda()
        real_label = Variable(Y_train_batch).cuda()
        disc_real_source, disc_real_class = aD(real_data)

        prediction = disc_real_class.data.max(1)[1]
        accuracy = ( float( prediction.eq(real_label.data).sum() ) / float(batch_size)) * 100.0

        disc_real_source = disc_real_source.mean()
        disc_real_class = criterion(disc_real_class, real_label)

        gradient_penalty = calc_gradient_penalty(aD, real_data, fake_data)

        disc_cost = disc_fake_source - disc_real_source + disc_real_class + disc_fake_class + gradient_penalty
        disc_cost.backward()

        optimizer_d.step()

        # Loss and accuracy within the current epoch.
        loss1.append(gradient_penalty.item())
        loss2.append(disc_fake_source.item())
        loss3.append(disc_real_source.item())
        loss4.append(disc_real_class.item())
        loss5.append(disc_fake_class.item())
        acc1.append(accuracy)

        if batch_idx % 50 == 0:
            print("[", epoch, batch_idx, "]", 
                    "%.2f" % np.mean(loss1), 
                    "%.2f" % np.mean(loss2), 
                    "%.2f" % np.mean(loss3), 
                    "%.2f" % np.mean(loss4), 
                    "%.2f" % np.mean(loss5), 
                    "%.2f" % np.mean(acc1))
    
    # Test the model after every epoch.
    aD.eval()
    with torch.no_grad():
        test_accu = []
        for batch_idx, (X_test_batch, Y_test_batch) in enumerate(testloader):
            X_test_batch, Y_test_batch= Variable(X_test_batch).cuda(), Variable(Y_test_batch).cuda()

            with torch.no_grad():
                _, output = aD(X_test_batch)

            prediction = output.data.max(1)[1] # first column has actual prob.
            accuracy = ( float( prediction.eq(Y_test_batch.data).sum() ) /float(batch_size))*100.0
            test_accu.append(accuracy)
            accuracy_test = np.mean(test_accu)
    
    print('Testing', accuracy_test, time.time()-start_time, "secs")

    # Save output.
    print("Saving output...")
    with torch.no_grad():
        aG.eval()
        samples = aG(save_noise)
        samples = samples.data.cpu().numpy()
        samples += 1.0
        samples /= 2.0
        samples = samples.transpose(0,2,3,1)
        aG.train()

    fig = plot(samples)
    plt.savefig('output/%s.png' % str(epoch).zfill(3), bbox_inches='tight')
    plt.close(fig)

    if (epoch + 1) % 1 == 0:
        torch.save(aG,'tempG.model')
        torch.save(aD,'tempD.model')


print("==> Training finished. Saving final model...")
torch.save(aG,'generator.model')
torch.save(aD,'discriminator.model')