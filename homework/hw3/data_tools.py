import torch
import torchvision
import torchvision.transforms as transforms


def data_loader_and_transformer(root_path):
    """Utils for loading and preprocessing data
    Args:
        root_path(string): the path to download/fetch the data
    Returns:
        train_data_loader(iterator)
        test_data_loader(iterator) 
    """

    # Data augmentation.
    # See https://github.com/kuangliu/pytorch-cifar/issues/19 for the normalization.
    train_data_transform = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261)),
    ])
    test_data_tranform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261)),
    ])

    # Data loader.
    train_dataset = torchvision.datasets.CIFAR10(
        root=root_path,
        train=True,
        download=True,
        transform=train_data_transform
    )
    train_data_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=128,
        shuffle=True,
        num_workers=2
    )

    test_dataset = torchvision.datasets.CIFAR10(
        root=root_path,
        train=False,
        download=True,
        transform=test_data_tranform
    )
    test_data_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=100,
        shuffle=False,
        num_workers=2
    )

    return train_data_loader, test_data_loader

