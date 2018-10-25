import torch
import torchvision
import torchvision.transforms as transforms
import multiprocessing

def data_loader_and_transformer(batch_size):
    transform_train = transforms.Compose([
        transforms.RandomResizedCrop(32, scale=(0.7, 1.0), ratio=(1.0,1.0)),
        transforms.ColorJitter(
                brightness=0.1*torch.randn(1),
                contrast=0.1*torch.randn(1),
                saturation=0.1*torch.randn(1),
                hue=0.1*torch.randn(1)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])

    transform_test = transforms.Compose([
        transforms.CenterCrop(32),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])

    workers = multiprocessing.cpu_count()
    print("\tnumber of workers: {}".format(workers))

    trainset = torchvision.datasets.CIFAR10(
        root='./', train=True, download=True, transform=transform_train
    )
    trainloader = torch.utils.data.DataLoader(
        trainset, 
        batch_size=batch_size, 
        shuffle=True, 
        num_workers=workers
    )

    testset = torchvision.datasets.CIFAR10(
        root='./', train=False, download=False, transform=transform_test
    )
    testloader = torch.utils.data.DataLoader(
        testset, 
        batch_size=batch_size, 
        shuffle=False, 
        num_workers=workers
    )

    return trainloader, testloader