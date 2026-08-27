import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms


# CIFAR-10 channel stats (precomputed on the training set)
_CIFAR10_MEAN = [0.4914, 0.4822, 0.4465]
_CIFAR10_STD = [0.2470, 0.2435, 0.2616]

# Fashion-MNIST is grayscale, so different stats
_FMNIST_MEAN = [0.2860]
_FMNIST_STD = [0.3530]


def get_transforms(dataset: str = "cifar10", train: bool = True) -> transforms.Compose:
    if dataset == "cifar10":
        mean, std = _CIFAR10_MEAN, _CIFAR10_STD
        if train:
            return transforms.Compose([
                transforms.RandomHorizontalFlip(),
                transforms.RandomCrop(32, padding=4),
                # TODO: add cutout or mixup for better regularization
                transforms.ToTensor(),
                transforms.Normalize(mean, std),
            ])
        return transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ])

    elif dataset == "fashion_mnist":
        mean, std = _FMNIST_MEAN, _FMNIST_STD
        if train:
            return transforms.Compose([
                transforms.RandomHorizontalFlip(),
                transforms.Grayscale(num_output_channels=3),  # model expects 3 channels
                transforms.ToTensor(),
                transforms.Normalize(mean * 3, std * 3),
            ])
        return transforms.Compose([
            transforms.Grayscale(num_output_channels=3),
            transforms.ToTensor(),
            transforms.Normalize(mean * 3, std * 3),
        ])

    else:
        raise ValueError(f"Unsupported dataset: {dataset}. Use 'cifar10' or 'fashion_mnist'")


def get_dataloaders(
    data_dir: str,
    batch_size: int = 64,
    num_workers: int = 2,
    dataset: str = "cifar10",
) -> tuple[DataLoader, DataLoader]:
    # cap workers to avoid issues on machines with fewer cores
    num_workers = min(num_workers, 4)
    use_pin_memory = torch.cuda.is_available()

    if dataset == "cifar10":
        train_ds = datasets.CIFAR10(
            root=data_dir, train=True, download=True,
            transform=get_transforms("cifar10", train=True),
        )
        val_ds = datasets.CIFAR10(
            root=data_dir, train=False, download=True,
            transform=get_transforms("cifar10", train=False),
        )
    elif dataset == "fashion_mnist":
        train_ds = datasets.FashionMNIST(
            root=data_dir, train=True, download=True,
            transform=get_transforms("fashion_mnist", train=True),
        )
        val_ds = datasets.FashionMNIST(
            root=data_dir, train=False, download=True,
            transform=get_transforms("fashion_mnist", train=False),
        )
    else:
        raise ValueError(f"Unsupported dataset: {dataset}")

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=use_pin_memory,
        persistent_workers=num_workers > 0,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=use_pin_memory,
        persistent_workers=num_workers > 0,
    )

    return train_loader, val_loader
