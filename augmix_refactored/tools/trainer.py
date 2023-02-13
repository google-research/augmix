import torch
import torch.backends.cudnn as cudnn
import torch.nn.functional as F
from tqdm import tqdm

from augmix_refactored.config import Config

def train(net, train_loader, optimizer, scheduler, config: Config, logging, epoch):
    """Train for one epoch."""
    net.train()
    loss_ema = 0.
    iter = 0
    with tqdm(train_loader, unit="batch", disable=config.disable_tqdm) as tepoch:
        for images, targets in tepoch:
        #for i, (images, targets) in enumerate(train_loader):
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
            tepoch.set_description(f"Epoch {epoch}")
            tepoch.set_postfix(train_loss=loss_ema)
            if config.disable_tqdm and iter % config.print_freq == 0:
                logging.info('Train Loss {:.3f}'.format(loss_ema))
            iter +=1

    return loss_ema