import torch
from torchvision import datasets
from torchvision import transforms
import __augmentations
from augmix_refactored.augmentor import Augmentor
from augmix_refactored.config import Config
from augmix_refactored.aug_mix_dataset import AugMixDataset


def cifar_transforms():
    train_transform = transforms.Compose(
        [transforms.RandomHorizontalFlip(),
         transforms.RandomCrop(32, padding=4)])
    preprocess = transforms.Compose(
        [transforms.ToTensor(),
         transforms.Normalize([0.5] * 3, [0.5] * 3)])
    test_transform = preprocess
    return train_transform, test_transform, preprocess

def get_data(config: Config):
    # Load datasets
    train_transform, test_transform, preprocess = cifar_transforms()

    if config.dataset == 'cifar10':
        train_data = datasets.CIFAR10(
            './data/cifar', train=True, transform=train_transform, download=True)
        test_data = datasets.CIFAR10(
            './data/cifar', train=False, transform=test_transform, download=True)
        base_c_path = './data/cifar/CIFAR-10-C/'
        num_classes = 10
    else:
        train_data = datasets.CIFAR100(
            './data/cifar', train=True, transform=train_transform, download=True)
        test_data = datasets.CIFAR100(
            './data/cifar', train=False, transform=test_transform, download=True)
        base_c_path = './data/cifar/CIFAR-100-C/'
        num_classes = 100
    
    # Creating Augmentor
    aug = Augmentor(config)    

    train_data = AugMixDataset(train_data, preprocess, augmentor=aug, no_jsd=config.no_jsd)
    train_loader = torch.utils.data.DataLoader(
        train_data,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=config.num_workers,
        pin_memory=True)

    test_loader = torch.utils.data.DataLoader(
        test_data,
        batch_size=config.eval_batch_size,
        shuffle=False,
        num_workers=config.num_workers,
        pin_memory=True)
    
    return train_loader, test_loader, base_c_path, num_classes