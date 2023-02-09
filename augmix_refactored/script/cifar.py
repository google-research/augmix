# Copyright 2019 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Main script to launch AugMix training on CIFAR-10/100.

Supports WideResNet, AllConv, ResNeXt models on CIFAR-10 and CIFAR-100 as well
as evaluation on CIFAR-10-C and CIFAR-100-C.

Example usage:
  `python cifar.py`
"""
from __future__ import print_function
import logging 
import argparse
import os
import shutil
import time
import logging

import __augmentations
from augmix_refactored.augmentor import Augmentor
from augmix_refactored.models import AllConvNet
from augmix_refactored.config import Config
from augmix_refactored.aug_mix_dataset import AugMixDataset
import numpy as np
from augmix_refactored.third_party.ResNeXt_DenseNet.models.densenet import densenet
from augmix_refactored.third_party.ResNeXt_DenseNet.models.resnext import resnext29
from augmix_refactored.third_party.WideResNet_pytorch.wideresnet import WideResNet

import torch
import torch.backends.cudnn as cudnn
import torch.nn.functional as F
from torchvision import datasets
from torchvision import transforms

CORRUPTIONS = [
    'gaussian_noise', 'shot_noise', 'impulse_noise', 'defocus_blur',
    'glass_blur', 'motion_blur', 'zoom_blur', 'snow', 'frost', 'fog',
    'brightness', 'contrast', 'elastic_transform', 'pixelate',
    'jpeg_compression'
]


def get_lr(step, total_steps, lr_max, lr_min):
    """Compute learning rate according to cosine annealing schedule."""
    return lr_min + (lr_max - lr_min) * 0.5 * (1 +
                                               np.cos(step / total_steps * np.pi))


def train(net, train_loader, optimizer, scheduler, config: Config):
    """Train for one epoch."""
    net.train()
    loss_ema = 0.
    for i, (images, targets) in enumerate(train_loader):
        optimizer.zero_grad()

        if config.no_jsd:
            images = images.cuda()
            targets = targets.cuda()
            logits = net(images)
            loss = F.cross_entropy(logits, targets)
        else:
            images_all = torch.cat(images, 0).cuda()
            targets = targets.cuda()
            logits_all = net(images_all)
            logits_clean, logits_aug1, logits_aug2 = torch.split(
                logits_all, images[0].size(0))

            # Cross-entropy is only computed on clean images
            loss = F.cross_entropy(logits_clean, targets)

            p_clean, p_aug1, p_aug2 = F.softmax(
                logits_clean, dim=1), F.softmax(
                    logits_aug1, dim=1), F.softmax(
                        logits_aug2, dim=1)

            # Clamp mixture distribution to avoid exploding KL divergence
            p_mixture = torch.clamp(
                (p_clean + p_aug1 + p_aug2) / 3., 1e-7, 1).log()
            loss += 12 * (F.kl_div(p_mixture, p_clean, reduction='batchmean') +
                          F.kl_div(p_mixture, p_aug1, reduction='batchmean') +
                          F.kl_div(p_mixture, p_aug2, reduction='batchmean')) / 3.

        loss.backward()
        optimizer.step()
        scheduler.step()
        loss_ema = loss_ema * 0.9 + float(loss) * 0.1
        if i % config.print_freq == 0:
            logging.info('Train Loss {:.3f}'.format(loss_ema))

    return loss_ema


def test(net, test_loader):
    """Evaluate network on given dataset."""
    net.eval()
    total_loss = 0.
    total_correct = 0
    with torch.no_grad():
        for images, targets in test_loader:
            images, targets = images.cuda(), targets.cuda()
            logits = net(images)
            loss = F.cross_entropy(logits, targets)
            pred = logits.data.max(1)[1]
            total_loss += float(loss.data)
            total_correct += pred.eq(targets.data).sum().item()

    return total_loss / len(test_loader.dataset), total_correct / len(
        test_loader.dataset)


def test_c(net, test_data, base_path, config: Config):
    """Evaluate network on given corrupted dataset."""
    corruption_accs = []
    for corruption in CORRUPTIONS:
        # Reference to original data is mutated
        test_data.data = np.load(base_path + corruption + '.npy')
        test_data.targets = torch.LongTensor(np.load(base_path + 'labels.npy'))

        test_loader = torch.utils.data.DataLoader(
            test_data,
            batch_size=config.eval_batch_size,
            shuffle=False,
            num_workers=config.num_workers,
            pin_memory=True)

        test_loss, test_acc = test(net, test_loader)
        corruption_accs.append(test_acc)
        logging.info('{}\n\tTest Loss {:.3f} | Test Error {:.3f}'.format(
            corruption, test_loss, 100 - 100. * test_acc))

    return np.mean(corruption_accs)


def main():

    parser = argparse.ArgumentParser(
        description='Trains a CIFAR Classifier',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    
    parser = Config.get_parser(parser)
    args = parser.parse_args()
    
    config: Config = None
    if args.config_path:
        config = Config.load_from_yaml(args.config_path)
        config.apply_parsed_args(args)
    else:
        config = Config.from_parsed_args(args)

    torch.manual_seed(1)
    np.random.seed(1)

    # Load datasets
    train_transform = transforms.Compose(
        [transforms.RandomHorizontalFlip(),
         transforms.RandomCrop(32, padding=4)])
    preprocess = transforms.Compose(
        [transforms.ToTensor(),
         transforms.Normalize([0.5] * 3, [0.5] * 3)])
    test_transform = preprocess

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

    # Create model
    if config.model == 'densenet':
        net = densenet(num_classes=num_classes)
    elif config.model == 'wrn':
        net = WideResNet(config.layers, num_classes,
                         config.widen_factor, config.droprate)
    elif config.model == 'allconv':
        net = AllConvNet(num_classes)
    elif config.model == 'resnext':
        net = resnext29(num_classes=num_classes)

    optimizer = torch.optim.SGD(
        net.parameters(),
        config.learning_rate,
        momentum=config.momentum,
        weight_decay=config.weight_decay,
        nesterov=True)

    # Distribute model across all visible GPUs
    net = torch.nn.DataParallel(net).cuda()
    cudnn.benchmark = True

    start_epoch = 0

    if config.resume_path:
        if os.path.isfile(config.resume_path):
            checkpoint = torch.load(config.resume_path)
            start_epoch = checkpoint['epoch'] + 1
            best_acc = checkpoint['best_acc']
            net.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            logging.info('Model restored from epoch:', start_epoch)

    if config.evaluate:
        # Evaluate clean accuracy first because test_c mutates underlying data
        test_loss, test_acc = test(net, test_loader)
        logging.info('Clean\n\tTest Loss {:.3f} | Test Error {:.2f}'.format(
            test_loss, 100 - 100. * test_acc))

        test_c_acc = test_c(net, test_data, base_c_path)
        logging.info('Mean Corruption Error: {:.3f}'.format(
            100 - 100. * test_c_acc))
        return

    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        lr_lambda=lambda step: get_lr(  # pylint: disable=g-long-lambda
            step,
            config.epochs * len(train_loader),
            1,  # lr_lambda computes multiplicative factor
            1e-6 / config.learning_rate))

    if not os.path.exists(config.save_folder):
        os.makedirs(config.save_folder)
    if not os.path.isdir(config.save_folder):
        raise Exception('%s is not a dir' % config.save_folder)

    log_path = os.path.join(config.save_folder,
                            config.dataset + '_' + config.model + '_training_log.csv')
    with open(log_path, 'w') as f:
        f.write('epoch,time(s),train_loss,test_loss,test_error(%)\n')

    best_acc = 0
    logging.info('Beginning training from epoch:', start_epoch + 1)
    for epoch in range(start_epoch, config.epochs):
        begin_time = time.time()

        train_loss_ema = train(net, train_loader, optimizer, scheduler, config)
        test_loss, test_acc = test(net, test_loader)

        is_best = test_acc > best_acc
        best_acc = max(test_acc, best_acc)
        checkpoint = {
            'epoch': epoch,
            'dataset': config.dataset,
            'model': config.model,
            'state_dict': net.state_dict(),
            'best_acc': best_acc,
            'optimizer': optimizer.state_dict(),
        }

        save_path = os.path.join(config.save_folder, 'checkpoint.pth.tar')
        torch.save(checkpoint, save_path)
        if is_best:
            shutil.copyfile(save_path, os.path.join(
                config.save_folder, 'model_best.pth.tar'))

        with open(log_path, 'a') as f:
            f.write('%03d,%05d,%0.6f,%0.5f,%0.2f\n' % (
                (epoch + 1),
                time.time() - begin_time,
                train_loss_ema,
                test_loss,
                100 - 100. * test_acc,
            ))

        logging.info(
            'Epoch {0:3d} | Time {1:5d} | Train Loss {2:.4f} | Test Loss {3:.3f} |'
            ' Test Error {4:.2f}'
            .format((epoch + 1), int(time.time() - begin_time), train_loss_ema,
                    test_loss, 100 - 100. * test_acc))

    test_c_acc = test_c(net, test_data, base_c_path)
    logging.info('Mean Corruption Error: {:.3f}'.format(
        100 - 100. * test_c_acc))

    with open(log_path, 'a') as f:
        f.write('%03d,%05d,%0.6f,%0.5f,%0.2f\n' %
                (config.epochs + 1, 0, 0, 0, 100 - 100 * test_c_acc))


if __name__ == '__main__':
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    main()
