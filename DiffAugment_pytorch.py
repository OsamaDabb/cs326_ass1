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
    # Manually augment the brightness of the images using direct scaling

    jittered_image = torch.zeros_like(x)

    for idx, img in enumerate(x):

        brightness_factor = random.uniform(0.5, 1.5)

        # Apply the brightness factor
        jittered_image[idx] = torch.clamp(img * brightness_factor, 0, 1)

    return jittered_image




def augment_saturation(x):
    # Uses torch ColorJitter to uniformly adjust saturation between 0-50%

    return x


def augment_contrast(x):
    # Uses torch ColorJitter to uniformly adjust contrast between 0-10%

    jittered_image = torch.zeros_like(x)

    for idx, img in enumerate(x):
        contrast_factor = random.uniform(0.5, 1.5)

        # Apply the brightness factor
        jittered_image[idx] = torch.clamp(img + (img - img.mean()) * contrast_factor, 0, 1)

    return jittered_image


def augment_translation(x, ratio=0.125):
    transform = torchvision.transforms.RandomAffine(degrees=0, translate=(ratio, ratio))
    to_pil = torchvision.transforms.ToPILImage()
    to_tens = torchvision.transforms.ToTensor()

    # Apply the transformation to each image in the batch
    translated_images = torch.stack([to_tens(
                                        transform(to_pil(img.cpu()))) for img in x])

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
