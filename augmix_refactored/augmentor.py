from types import MethodType
from typing import List
import numpy as np
from PIL import Image, ImageOps, ImageEnhance
from augmix_refactored.tools.helper import int_parameter, float_parameter, sample_level
import torch
from augmix_refactored.config import Config

# TODO Add Docstrings and correct method return types to all functions.

class Augmentor:
    """Class providing augmentation functions."""

    config: Config
    """The config within the training."""

    def __call__(self, image, preprocess):
        """Perform AugMix augmentations and compute mixture.

        Args:
            image: PIL.Image input image
            preprocess: Preprocessing function which should return a torch tensor.

        Returns:
            mixed: Augmented and mixed image.
        """
        args = self.config
        aug_list = self.get_augmentations()
        if args.all_ops:
            aug_list = self.get_augmentations_all()

        ws = np.float32(np.random.dirichlet([1] * args.mixture_width))
        m = np.float32(np.random.beta(1, 1))

        mix = torch.zeros_like(preprocess(image))
        for i in range(args.mixture_width):
            image_aug = image.copy()
            depth = args.mixture_depth if args.mixture_depth > 0 else np.random.randint(
                1, 4)
            for _ in range(depth):
                op = np.random.choice(aug_list)
                image_aug = op(image_aug, args.augmentation_severity)
                # Preprocessing commutes since all coefficients are convex
                mix += ws[i] * preprocess(image_aug)

        mixed = (1 - m) * preprocess(image) + m * mix
        return mixed



    def __init__(self, config: Config) -> None:
        self.config = config


    def autocontrast(self, pil_img: Image, _):
        return ImageOps.autocontrast(pil_img)

    def equalize(self, pil_img: Image, _):
        return ImageOps.equalize(pil_img)

    def posterize(self, pil_img: Image, level):
        level = int_parameter(sample_level(level), 4)
        return ImageOps.posterize(pil_img, 4 - level)

    def rotate(self, pil_img: Image, level):
        degrees = int_parameter(sample_level(level), 30)
        if np.random.uniform() > 0.5:
            degrees = -degrees
        return pil_img.rotate(degrees, resample=Image.BILINEAR)

    def solarize(self, pil_img: Image, level):
        level = int_parameter(sample_level(level), 256)
        return ImageOps.solarize(pil_img, 256 - level)

    def shear_x(self, pil_img: Image, level):
        level = float_parameter(sample_level(level), 0.3)
        if np.random.uniform() > 0.5:
            level = -level
        return pil_img.transform((self.config.image_size, self.config.image_size),
                                 Image.AFFINE, (1, level, 0, 0, 1, 0),
                                 resample=Image.BILINEAR)

    def shear_y(self, pil_img: Image, level):
        level = float_parameter(sample_level(level), 0.3)
        if np.random.uniform() > 0.5:
            level = -level
        return pil_img.transform((self.config.image_size, self.config.image_size),
                                 Image.AFFINE, (1, 0, 0, level, 1, 0),
                                 resample=Image.BILINEAR)

    def translate_x(self, pil_img: Image, level):
        level = int_parameter(sample_level(level), self.config.image_size / 3)
        if np.random.random() > 0.5:
            level = -level
        return pil_img.transform((self.config.image_size, self.config.image_size),
                                 Image.AFFINE, (1, 0, level, 0, 1, 0),
                                 resample=Image.BILINEAR)

    def translate_y(self, pil_img: Image, level):
        level = int_parameter(sample_level(level), self.config.image_size / 3)
        if np.random.random() > 0.5:
            level = -level
        return pil_img.transform((self.config.image_size, self.config.image_size),
                                 Image.AFFINE, (1, 0, 0, 0, 1, level),
                                 resample=Image.BILINEAR)

        # operation that overlaps with ImageNet-C's test set

    def color(self, pil_img: Image, level):
            level = float_parameter(sample_level(level), 1.8) + 0.1
            return ImageEnhance.Color(pil_img).enhance(level)

        # operation that overlaps with ImageNet-C's test set

    def contrast(self, pil_img: Image, level):
            level = float_parameter(sample_level(level), 1.8) + 0.1
            return ImageEnhance.Contrast(pil_img).enhance(level)

        # operation that overlaps with ImageNet-C's test set

    def brightness(self, pil_img: Image, level):
            level = float_parameter(sample_level(level), 1.8) + 0.1
            return ImageEnhance.Brightness(pil_img).enhance(level)

        # operation that overlaps with ImageNet-C's test set

    def sharpness(self, pil_img: Image, level):
            level = float_parameter(sample_level(level), 1.8) + 0.1
            return ImageEnhance.Sharpness(pil_img).enhance(level)

    def get_augmentations(self) -> List[MethodType]:
        """Getting the basic augmentation methods.

        Returns
        -------
        List[MethodType]
            A list of basic invokable augmentation methods.
        """
        return [
            self.autocontrast, self.equalize, self.posterize, self.rotate, self.solarize, self.shear_x, self.shear_y,
            self.translate_x, self.translate_y
        ]
    
    def get_augmentations_all(self) -> List[MethodType]:
        """A list of all invokable augmentation methods.

        Returns
        -------
        List[MethodType]
            A list of all augmentation methods.
        """
        return [
            self.autocontrast, self.equalize, self.posterize, self.rotate, self.solarize, self.shear_x, self.shear_y,
            self.translate_x, self.translate_y, self.color, self.contrast, self.brightness, self.sharpness
        ]
