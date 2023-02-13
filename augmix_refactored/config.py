from dataclasses import dataclass, field
from typing import Literal, Optional
from augmix_refactored.tools.argparser_mixin import ArgparserMixin
from augmix_refactored.tools.simple_yaml_mixin import SimpleYamlMixin

@dataclass
class Config(ArgparserMixin, SimpleYamlMixin):
    """Basic config class for training with cifar."""

    dataset: Literal['cifar10', 'cifar100'] = "cifar10"
    """Describing which dataset should be used"""

    model: Literal['wrn', 'allconv', 'densenet', 'resnext'] = "wrn"
    """The architecture which should be used."""

    epochs: int = 100
    """Number of epochs to train."""

    learning_rate: float = 0.1
    """Initial learning rate."""

    batch_size: int = 128
    """The batch size used for training"""

    eval_batch_size: int = 1000
    """Batch size used for evaluation."""

    momentum: float = 0.9
    """Momentum for the SGD Optimizer."""

    weight_decay: float = 0.0005
    """Weight decay (L2 penalty) for the optimizer."""

    layers: int = 40
    """Total number of layers."""

    widen_factor: int = 2
    """Widen factor"""

    droprate: float = 0.0
    """Dropout probability"""

    mixture_width: int = 3
    """Number of augmentation chains to mix per augmented example."""

    mixture_depth: int = -1
    """Depth of augmentation chains. -1 denotes stochastic depth in [1, 3]"""

    augmentation_severity: int = 3
    """Severity of base augmentation operators."""

    no_jsd: bool = False
    """Turn off JSD consistency loss."""

    all_ops: bool = False
    """Turn on all operations (+brightness,contrast,color,sharpness)."""

    save_folder: str = "./snapshots"
    """Folder to save checkpoints."""

    resume_path: str = ""
    """Checkpoint path for resume / test."""

    evaluate: bool = False
    """Eval only."""

    print_freq: int = 50
    """Training loss print frequency (batches)."""

    num_workers: int = 4
    """Number of pre-fetching threads."""

    config_path: Optional[str] = None
    """Config path to load."""

    image_size: int = 32
    """Image size for images in the dataset."""

    disable_tqdm: bool = False
    """When using TQDM with slurm, the slurm.err file has a progress bar for every single increment. Thus progress bar can be turned off if need be."""