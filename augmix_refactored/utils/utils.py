import torch
import numpy as np
import logging

from augmix_refactored.config import Config

from augmix_refactored.third_party.ResNeXt_DenseNet.models.densenet import densenet
from augmix_refactored.third_party.ResNeXt_DenseNet.models.resnext import resnext29
from augmix_refactored.third_party.WideResNet_pytorch.wideresnet import WideResNet
from augmix_refactored.models import AllConvNet

def get_lr(step, total_steps, lr_max, lr_min):
    """Compute learning rate according to cosine annealing schedule."""
    return lr_min + (lr_max - lr_min) * 0.5 * (1 +
                                               np.cos(step / total_steps * np.pi))


def setup_logger(log_path):
    logger_path = log_path.replace('training_log.csv', 'log.log')
    logging.basicConfig(filename=logger_path, filemode='a')
    logger_name = "logger"
    logger = logging.getLogger(logger_name)
    logger.setLevel(logging.INFO)
    handler = logging.StreamHandler()
    fmt = "[%(asctime)s %(levelname)s %(filename)s line %(lineno)d %(process)d] %(message)s"
    handler.setFormatter(logging.Formatter(fmt))
    logger.addHandler(handler)
    return logger

def get_model(config: Config, num_classes):
    if config.model == 'densenet':
        net = densenet(num_classes=num_classes)
    elif config.model == 'wrn':
        net = WideResNet(config.layers, num_classes,
                         config.widen_factor, config.droprate)
    elif config.model == 'allconv':
        net = AllConvNet(num_classes)
    elif config.model == 'resnext':
        net = resnext29(num_classes=num_classes)
    return net

def get_optimizer(config: Config, net=None):
    optimizer = torch.optim.SGD(
        net.parameters(),
        config.learning_rate,
        momentum=config.momentum,
        weight_decay=config.weight_decay,
        nesterov=True)
    return optimizer

def get_lr_scheduler(config: Config, optimizer, len_train_loader=50000):
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        lr_lambda=lambda step: get_lr(  # pylint: disable=g-long-lambda
            step,
            config.epochs * len_train_loader,
            1,  # lr_lambda computes multiplicative factor
            1e-6 / config.learning_rate))
    return scheduler