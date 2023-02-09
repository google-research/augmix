import torch
from .augmentor import Augmentor

class AugMixDataset(torch.utils.data.Dataset):
  """Dataset wrapper to perform AugMix augmentation."""

  def __init__(self, dataset, preprocess, augmentor:Augmentor, no_jsd=False):
    self.dataset = dataset
    self.preprocess = preprocess
    self.no_jsd = no_jsd
    self.augmentor = augmentor

  def __getitem__(self, i):
    x, y = self.dataset[i]
    if self.no_jsd:
      return self.augmentor(x, self.preprocess), y
    else:
      im_tuple = (self.preprocess(x), self.augmentor(x, self.preprocess),
                  self.augmentor(x, self.preprocess))
      return im_tuple, y

  def __len__(self):
    return len(self.dataset)