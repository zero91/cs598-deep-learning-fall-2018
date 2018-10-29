import torch
import torch.nn as nn
from torch import autograd
from torch.autograd import Variable
import torchvision
import torchvision.transforms as transforms

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import os

import multiprocessing
import numpy as np

import argparse
def str2bool(v):
    return v.lower() in ("yes", "true", "t", "1")

parser = argparse.ArgumentParser(description="Part2: Visualization")

# Visualization options.
parser.add_argument("--perturb_images", default=True, type=str2bool, 
                    help="section 1: pertube real image")
parser.add_argument("--max_classification_output", default=False, type=str2bool, 
                    help="section 2: maximizing classification output")
parser.add_argument("--max_features", default=False, type=str2bool, 
                    help="section 2: maximizing features at various layers")

args = parser.parse_args()

"""
python3.6 --perturb_images False --max_classification_output False --max_features True
"""

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

def main():
    # Load data and model.
    batch_size = 128

    print("==> Loading data...")
    transform_test = transforms.Compose([
        transforms.CenterCrop(32),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])

    testset = torchvision.datasets.CIFAR10(
        root='./', 
        train=False, 
        download=False, 
        transform=transform_test)

    workers = multiprocessing.cpu_count()
    print("\tnumber of workers: {}".format(workers))
    testloader = torch.utils.data.DataLoader(
        testset, 
        batch_size=batch_size, 
        shuffle=False, 
        num_workers=workers)
    # testloader = enumerate(testloader)
    testloader = iter(testloader)

    print("==> Loding discriminator model trained without the generator...")
    model = torch.load('cifar10.model')
    model.cuda()
    model.eval()

    if args.max_classification_output or args.max_features:
        print("==> Loding discriminator model trained with the generator...")
        model2 = torch.load('tempD.model')
        model2.cuda()
        model2.eval()

    # Grab a sample batch from the test dataset and create an alternative label.
    print("==> Sampling a batch...")
    # batch_idx, (X_batch, Y_batch) = testloader.next()
    X_batch, Y_batch = testloader.next()
    X_batch = Variable(X_batch,requires_grad=True).cuda()
    Y_batch_alternate = (Y_batch + 1) % 10
    Y_batch_alternate = Variable(Y_batch_alternate).cuda()
    Y_batch = Variable(Y_batch).cuda()

    # Save the first 100 real images.
    samples = X_batch.data.cpu().numpy()
    samples += 1.0
    samples /= 2.0
    samples = samples.transpose(0,2,3,1)

    fig = plot(samples[0:100])
    plt.savefig('visualization/real_images.png', bbox_inches='tight')
    plt.close(fig)

    if args.perturb_images:
        print("==> Section 1: Pertube real images")
        
        # Get the output from the fc10 layer
        print("=> Evaluating real images...")
        _, output = model(X_batch)
        prediction = output.data.max(1)[1] # first column has actual prob.
        accuracy = ( float( prediction.eq(Y_batch.data).sum() ) /float(batch_size)) * 100.0
        print("Accuracy for real images:", accuracy)

        # Slightly jitter all input images (loss is based on the alternative label).
        print("=> Evaluating fake images...")
        criterion = nn.CrossEntropyLoss(reduce=False)
        loss = criterion(output, Y_batch_alternate)

        gradients = torch.autograd.grad(outputs=loss, inputs=X_batch,
                                grad_outputs=torch.ones(loss.size()).cuda(),
                                create_graph=True, retain_graph=False, only_inputs=True)[0]

        # Save gradient jitter.
        gradient_image = gradients.data.cpu().numpy()
        gradient_image = ((gradient_image - np.min(gradient_image)) /
                        (np.max(gradient_image)-np.min(gradient_image)))
        gradient_image = gradient_image.transpose(0, 2, 3, 1)
        fig = plot(gradient_image[0:100])
        plt.savefig('visualization/gradient_image.png', bbox_inches='tight')
        plt.close(fig)

        # Jitter input image.
        gradients[gradients>0.0] = 1.0
        gradients[gradients<0.0] = -1.0

        gain = 8.0
        X_batch_modified = X_batch - gain*0.007843137*gradients
        X_batch_modified[X_batch_modified>1.0] = 1.0
        X_batch_modified[X_batch_modified<-1.0] = -1.0

        # Evaluate new fake images.
        _, output = model(X_batch_modified)
        prediction = output.data.max(1)[1] # first column has actual prob.
        accuracy = ( float( prediction.eq(Y_batch.data).sum() ) /float(batch_size))*100.0
        print("Accuracy for fake images:", accuracy)

        # Save fake images.
        samples = X_batch_modified.data.cpu().numpy()
        samples += 1.0
        samples /= 2.0
        samples = samples.transpose(0,2,3,1)

        fig = plot(samples[0:100])
        plt.savefig('visualization/jittered_images.png', bbox_inches='tight')
        plt.close(fig)
    
    if args.max_classification_output:
        print("==> Section 2: Synthetic Images Maximizing Classification Output")

        # Get mean image and make 10 copies.
        X = X_batch.mean(dim=0)
        X = X.repeat(10, 1, 1, 1)

        # Make a unique label for each copy.
        Y = torch.arange(10).type(torch.int64)
        Y = Variable(Y).cuda()

        # Generate synthetic images for both models in part 1.
        models = [model, model2]
        types = ["D trained without G", "D trained with G"]
        saved_names = ["no_g", "w_g"]
        lr = 0.1
        weight_decay = 0.001

        for model_idx, model in enumerate(models):
            print("=> Generating images for {}".format(types[model_idx]))
            out_file = open('visualization/max_class_out_{}.txt'
                            .format(saved_names[model_idx]), 'w')

            for i in range(200):
                _, output = model(X)

                loss = -output[torch.arange(10).type(torch.int64),torch.arange(10).type(torch.int64)]
                gradients = torch.autograd.grad(outputs=loss, inputs=X,
                                        grad_outputs=torch.ones(loss.size()).cuda(),
                                        create_graph=True, retain_graph=False, only_inputs=True)[0]

                prediction = output.data.max(1)[1] # first column has actual prob.
                accuracy = ( float( prediction.eq(Y.data).sum() ) /float(10.0))*100.0
                print(i, accuracy, -loss)
                out_file.write('{} {} {}\n'.format(i, accuracy, -loss))

                X = X - lr*gradients.data - weight_decay*X.data*torch.abs(X.data)
                X[X>1.0] = 1.0
                X[X<-1.0] = -1.0
            
            out_file.close()

            # Save new images.
            samples = X.data.cpu().numpy()
            samples += 1.0
            samples /= 2.0
            samples = samples.transpose(0,2,3,1)

            fig = plot(samples)
            plt.savefig('visualization/max_class_{}.png'.format(saved_names[model_idx]), 
                        bbox_inches='tight')
            plt.close(fig)
    
    if args.max_features:
        print("==> Section 3: Synthetic Features Maximizing Features")

        X = X_batch.mean(dim=0)
        X = X.repeat(batch_size,1,1,1)

        Y = torch.arange(batch_size).type(torch.int64)
        Y = Variable(Y).cuda()

        # Generate synthetic images for both models in part 1.
        models = [model, model2]
        types = ["D trained without G", "D trained with G"]
        saved_names = ["no_g", "w_g"]
        layers = [4, 8]

        lr = 0.1
        weight_decay = 0.001

        for model_idx, model in enumerate(models):
            for layer in layers:
                print("=> Generating images for {} after layer {}"
                      .format(types[model_idx], layer))
                out_file = open('visualization/max_features_out_{}_l{}.txt'
                                .format(saved_names[model_idx], layer), 'w')
            
                for i in range(200):
                    model.set_extract_features(layer)
                    # _, output = model(X, Variable(torch.IntTensor(layer)))
                    output = model(X)

                    loss = -output[torch.arange(batch_size).type(torch.int64),torch.arange(batch_size).type(torch.int64)]
                    gradients = torch.autograd.grad(outputs=loss, inputs=X,
                                            grad_outputs=torch.ones(loss.size()).cuda(),
                                            create_graph=True, retain_graph=False, only_inputs=True)[0]

                    prediction = output.data.max(1)[1] # first column has actual prob.
                    accuracy = ( float( prediction.eq(Y.data).sum() ) /float(batch_size))*100.0
                    print(i, accuracy, -loss)
                    out_file.write('{} {} {}\n'.format(i, accuracy, -loss))
                    
                    X = X - lr*gradients.data - weight_decay*X.data*torch.abs(X.data)
                    X[X>1.0] = 1.0
                    X[X<-1.0] = -1.0
                
                out_file.close()

                # Save new images.
                samples = X.data.cpu().numpy()
                samples += 1.0
                samples /= 2.0
                samples = samples.transpose(0,2,3,1)

                fig = plot(samples[0:100])
                plt.savefig('visualization/max_features_{}_l{}.png'
                            .format(saved_names[model_idx], layer), bbox_inches='tight')
                plt.close(fig)


if __name__ == '__main__':
    main()