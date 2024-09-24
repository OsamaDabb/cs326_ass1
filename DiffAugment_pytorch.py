"""
Defining the DiffAugment augmentations.
Implement your differentiable augmentation functions here.
"""

import torch, torchvision
import torch.nn.functional as F
import random



def DiffAugment(x, policy="", channels_first=True):
    if policy:
        if not channels_first:
            x = x.permute(0, 3, 1, 2)
        for p in policy.split(","):
            for f in AUGMENT_FNS[p]:
                x = f(x)
        if not channels_first:
            x = x.permute(0, 2, 3, 1)
        x = x.contiguous()
    return x


def augment_brightness(x):
    # Uses torch ColorJitter to uniformly adjust brightness between 0-50%

    return torchvision.transforms.ColorJitter(x, brightness=0.5)


def augment_saturation(x):
    # Uses torch ColorJitter to uniformly adjust saturation between 0-50%
    return torchvision.transforms.ColorJitter(x, saturation=1)


def augment_contrast(x):
    # Uses torch ColorJitter to uniformly adjust contrast between 0-10%
    return torchvision.transforms.ColorJitter(x, contrast=0.5)


def augment_translation(x, ratio=0.125):
    # Use torch RandomAffine to translate image up to 10% in all four directions
    transform = torchvision.transforms.RandomAffine(degrees=0, translate=(1/8, 1/8))

    # Apply the transformation to each image in the batch
    translated_images = torch.stack([transform(img) for img in x])

    return translated_images


def augment_cutout(x, ratio=0.5):
    # assuming x: [B, C, L, W], define a region that is L*ratio x W*ratio,
    # finally, sets that region to zero

    l, w = x.shape[2], x.shape[3]
    mask_l, mask_w = int(l * ratio), int(w * ratio)

    mask_x, mask_y = random.randint(0, l - mask_l), random.randint(0, w - mask_w)

    x[:, :, mask_x:mask_x+mask_l, mask_y:mask_y+mask_w] = 0

    return x


AUGMENT_FNS = {
    "color": [augment_brightness, augment_saturation, augment_contrast],
    "translation": [augment_translation],
    "cutout": [augment_cutout],
}
