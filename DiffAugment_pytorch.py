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

    # add (rand() - 0.5) * x to x to randomly scale x by -50% to +50%,
    # effectively increasing or decreasing image brightness
    x = x + x * (torch.rand(x.size(0), 1, 1, 1, device=x.device) - 0.5)
    return x


def augment_saturation(x):

    # calculate mean of image across each channel
    mean_val = x.mean(dim=1, keepdim=True)

    # Scale the variance (x - mean_val) for each channel by constant 2*rand() to increase
    # relative color saturation across the image.
    x = (x - mean_val) * (torch.rand(x.size(0), 1, 1, 1, dtype=x.dtype, device=x.device) * 2) + mean_val
    return x


def augment_contrast(x):
    # calculate mean for each image, then multiply each value (x - mean_val) by rand(0.5-1.5), effectively
    # scaling the variance (and thus the contrast) of the image by rand(0.5-1.5)

    mean_val = x.mean(dim=[1,2,3], keep_dim=True)

    x = (x - mean_val) * (torch.rand(x.size(0), 1, 1, 1, device = x.device) + 0.5) + mean_val

    return x

def augment_translation(x, ratio=0.125):

    # Calculate the number of pixels to shift in x and y directions, based on ratio
    shift_x = int(x.size(2) * ratio + 0.5)  # Get shift in width
    shift_y = int(x.size(3) * ratio + 0.5)  # Get shift in height

    # Generate random translations within the bounds [-shift_x, shift_x] for each image in the batch
    translation_x = torch.randint(low=-shift_x, high=shift_x + 1, size=[x.size(0), 1, 1],
                                  device=x.device)  # Translation for width
    translation_y = torch.randint(low=-shift_y, high=shift_y + 1, size=[x.size(0), 1, 1],
                                  device=x.device)  # Translation for height

    # Create meshgrid for batch indices, width indices, and height indices
    batch_indices = torch.arange(x.size(0), dtype=torch.long, device=x.device)  # Batch size dimension
    height_indices = torch.arange(x.size(2), dtype=torch.long, device=x.device)  # Height dimension
    width_indices = torch.arange(x.size(3), dtype=torch.long, device=x.device)  # Width dimension
    grid_batch, grid_h, grid_w = torch.meshgrid(batch_indices, height_indices,
                                                width_indices)  # Create a grid of batch, height, width

    # Adjust grid for translation by adding the translations, and clamp the values to ensure they're within the valid range
    adjusted_grid_h = torch.clamp(grid_h + translation_x + 1, min=0, max=x.size(2) + 1)  # Clamping height grid
    adjusted_grid_w = torch.clamp(grid_w + translation_y + 1, min=0, max=x.size(3) + 1)  # Clamping width grid

    # Pad the input tensor 'x' by 1 pixel on all sides, except the batch and channel dimensions
    x_padded = F.pad(x, pad=[1, 1, 1, 1, 0, 0, 0, 0])  # Padding in the height and width dimensions only

    # Permute and apply the translations, rearranging the data and accessing it through the adjusted grid indices
    x_transformed = x_padded.permute(0, 2, 3, 1).contiguous()  # Rearrange dimensions to make height and width first
    x_transformed = x_transformed[grid_batch, adjusted_grid_h, adjusted_grid_w]  # Access values using the new grid
    x_transformed = x_transformed.permute(0, 3, 1,
                                          2).contiguous()  # Rearrange back to original batch, channel, height, width order

    # Return the transformed tensor 'x'
    return x_transformed


def augment_cutout(x, ratio=0.5):
    # Calculate the size of the cutout based on the ratio and input dimensions
    cutout_height = int(x.size(2) * ratio + 0.5)  # Cutout height
    cutout_width = int(x.size(3) * ratio + 0.5)  # Cutout width
    cutout_size = (cutout_height, cutout_width)  # Combined cutout size

    # Generate random offsets for the top-left corner of the cutout region within the valid range
    offset_x = torch.randint(0, x.size(2) + (1 - cutout_size[0] % 2), size=[x.size(0), 1, 1],
                             device=x.device)  # Offset for height
    offset_y = torch.randint(0, x.size(3) + (1 - cutout_size[1] % 2), size=[x.size(0), 1, 1],
                             device=x.device)  # Offset for width

    # Create a meshgrid of indices for batch size, cutout height, and cutout width
    batch_idx = torch.arange(x.size(0), dtype=torch.long, device=x.device)  # Batch index range
    cutout_h_idx = torch.arange(cutout_size[0], dtype=torch.long, device=x.device)  # Cutout height range
    cutout_w_idx = torch.arange(cutout_size[1], dtype=torch.long, device=x.device)  # Cutout width range
    grid_batch, grid_h, grid_w = torch.meshgrid(batch_idx, cutout_h_idx, cutout_w_idx)  # Generate meshgrid

    # Adjust the grid by shifting the cutout and clamp the values to ensure they stay within valid image dimensions
    adjusted_grid_h = torch.clamp(grid_h + offset_x - cutout_size[0] // 2, min=0,
                                  max=x.size(2) - 1)  # Adjusted height grid
    adjusted_grid_w = torch.clamp(grid_w + offset_y - cutout_size[1] // 2, min=0,
                                  max=x.size(3) - 1)  # Adjusted width grid

    # Create a mask of ones (same size as input) and set the cutout region to zero
    mask = torch.ones(x.size(0), x.size(2), x.size(3), dtype=x.dtype, device=x.device)  # All ones mask
    mask[grid_batch, adjusted_grid_h, adjusted_grid_w] = 0  # Set cutout region to zero

    # Apply the mask to the input tensor, cutting out the selected region
    x_cutout = x * mask.unsqueeze(1)  # Unsqueeze to match the input dimensions and apply mask

    return x_cutout  # Return the result with the cutout applied


AUGMENT_FNS = {
    "color": [augment_brightness, augment_saturation, augment_contrast],
    "translation": [augment_translation],
    "cutout": [augment_cutout],
}
