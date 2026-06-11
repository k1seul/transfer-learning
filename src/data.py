import random
import numpy as np
import torch
from torchvision import transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import Dataset, DataLoader

CUB_MEAN   = (0.485, 0.456, 0.406)
CUB_STD    = (0.229, 0.224, 0.225)
PAINT_MEAN = (0.7815, 0.7699, 0.7322)
PAINT_STD  = (0.2654, 0.2694, 0.2941)


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_transform(domain: str, augment: bool = False, heavy: bool = False):
    mean, std = (CUB_MEAN, CUB_STD) if domain == 'cub' else (PAINT_MEAN, PAINT_STD)

    if heavy:
        if domain == 'cub':
            # Simulate painting appearance: grayscale + blur + erasing
            ops = [
                transforms.RandomResizedCrop(224, scale=(0.2, 1.0)),
                transforms.RandomHorizontalFlip(),
                transforms.RandomApply(
                    [transforms.GaussianBlur(kernel_size=23, sigma=(0.1, 2.0))], p=0.5),
                transforms.RandomGrayscale(p=0.5),
                transforms.ColorJitter(brightness=0.5, contrast=0.5, saturation=0.7, hue=0.2),
                transforms.RandomRotation(20),
                transforms.ToTensor(),
                transforms.Normalize(mean, std),
                transforms.RandomErasing(p=0.3, scale=(0.02, 0.2)),
            ]
        else:
            # Paintings source: strong color diversity
            ops = [
                transforms.RandomResizedCrop(224, scale=(0.5, 1.0)),
                transforms.RandomHorizontalFlip(),
                transforms.ColorJitter(brightness=0.6, contrast=0.6, saturation=0.8, hue=0.2),
                transforms.RandomGrayscale(p=0.1),
                transforms.RandomRotation(15),
                transforms.ToTensor(),
                transforms.Normalize(mean, std),
            ]
    elif augment:
        ops = [
            transforms.Resize(224), transforms.CenterCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.1),
            transforms.RandomRotation(15),
            transforms.ToTensor(), transforms.Normalize(mean, std),
        ]
    else:
        ops = [transforms.Resize(224), transforms.CenterCrop(224),
               transforms.ToTensor(), transforms.Normalize(mean, std)]

    return transforms.Compose(ops)


def get_contrastive_transform(domain: str):
    mean, std = (CUB_MEAN, CUB_STD) if domain == 'cub' else (PAINT_MEAN, PAINT_STD)
    return transforms.Compose([
        transforms.RandomResizedCrop(224, scale=(0.2, 1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.1),
        transforms.RandomGrayscale(p=0.2),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])


class UnlabeledDataset(Dataset):
    def __init__(self, labeled_dataset):
        self.dataset = labeled_dataset

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        img, _ = self.dataset[idx]
        return img


class PseudoLabeledDataset(Dataset):
    """High-confidence target samples with threshold-based pseudo-labels."""
    def __init__(self, root, transform, indices, pseudo_labels):
        self.base          = ImageFolder(root=root, transform=transform)
        self.indices       = indices
        self.pseudo_labels = pseudo_labels

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, i):
        img, _ = self.base[self.indices[i]]
        return img, self.pseudo_labels[i]


class PseudoTargetDataset(Dataset):
    """Full target dataset with prototype-based pseudo-labels (SHOT)."""
    def __init__(self, root, transform, pseudo_labels):
        self.base          = ImageFolder(root=root, transform=transform)
        self.pseudo_labels = pseudo_labels

    def __len__(self):
        return len(self.base)

    def __getitem__(self, idx):
        img, true_label = self.base[idx]
        return img, int(self.pseudo_labels[idx]), true_label


def make_loader(ds, batch_size, shuffle, num_workers=4):
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle,
                      num_workers=num_workers, pin_memory=True)
