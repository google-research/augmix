# AugMix

<img align="center" src="assets/augmix.gif" width="750">

## Introduction

We propose AugMix, a data processing technique that is mixes augmented images,
makes the augmented images consistently embedded, and results in increased
robustness and improved calibration. AugMix does not require tuning to work
correctly, so, like random cropping or CutOut, AugMix enables plug-and-play data
augmentation. AugMix significantly improves robustness and uncertainty measures
on challenging image classification benchmarks, closing the gap between previous
methods and the best possible performance in some cases by more than half. With
AugMix, we obtain state-of-the-art on ImageNet-C, ImageNet-P and in uncertainty
estimation when the train and test distribution do not match.

## Pseudocode

<img align="center" src="assets/pseudocode.png" width="750">

## Contents

This directory includes a reference implementation in NumPy of the augmentation
method used in AugMix in `augment_and_mix.py`. The full AugMix method also adds
a Jensen-Shanon Divergence consistency loss to enforce consistent predictions
between two different augmentations of the input image and the clean image
itself.

We also include PyTorch re-implementations of AugMix on both CIFAR-10/100 and
ImageNet in `cifar.py` and `imagenet.py` respectively, which both support
training and evaluation on CIFAR-10/100-C and ImageNet-C.

## Requirements

*   numpy>=1.15.0
*   Pillow>=6.1.0
*   torch==1.2.0
*   torchvision==0.2.2

## Setup

1.  Install PyTorch and other required python libraries with:

    ```
    pip install -r requirements.txt
    ```

2.  Download CIFAR-10-C and CIFAR-100-C datasets with:

    ```
    mkdir -p ./data/cifar
    curl -O https://zenodo.org/record/2535967/files/CIFAR-10-C.tar
    curl -O https://zenodo.org/record/3555552/files/CIFAR-100-C.tar
    tar -xvf CIFAR-100-C.tar -C data/cifar/
    tar -xvf CIFAR-10-C.tar -C data/cifar/
    ```

3.  Download ImageNet-C with:

    ```
    mkdir -p ./data/imagenet/imagenet-c
    curl -O https://zenodo.org/record/2235448/files/blur.tar
    curl -O https://zenodo.org/record/2235448/files/digital.tar
    curl -O https://zenodo.org/record/2235448/files/noise.tar
    curl -O https://zenodo.org/record/2235448/files/weather.tar
    tar -xvf blur.tar -C data/imagenet/imagenet-c
    tar -xvf digital.tar -C data/imagenet/imagenet-c
    tar -xvf noise.tar -C data/imagenet/imagenet-c
    tar -xvf weather.tar -C data/imagenet/imagenet-c
    ```

## Usage

Training recipes used in our paper:

WRN: `python cifar.py`

AllConv: `python cifar.py -m allconv`

ResNeXt: `python cifar.py -m resnext -e 200`

DenseNet: `python cifar.py -m densenet -e 200 -wd 0.0001`

ResNet-50: `python imagenet.py <path/to/imagenet> <path/to/imagenet-c>`

## Pretrained weights

Pretrained weights for ResNet-50 trained with AugMix on ImageNet are available
[here](https://drive.google.com/file/d/1CeguHRKednBToq2Ai_cY73jXPbXCIY7v/view?usp=sharing).

This model was measured at 66.2 mCE and 77.06% top-1 accuracy.

## Citation

If you find this useful for your work, please consider citing

```
@article{hendrycks2020augmix,
  title={{AugMix}: A Simple Data Processing Method to Improve Robustness and Uncertainty},
  author={Hendrycks, Dan and Mu, Norman and Cubuk, Ekin D. and Zoph, Barret and Gilmer, Justin and Lakshminarayanan, Balaji},
  journal={arXiv},
  year={2020}
}
```
