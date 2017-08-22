import numpy as np
from skimage import transform


def crop_center(img, crop_size):
    """Center crop images."""
    x, y = img.shape[:2]
    cx, cy = crop_size
    startx = x // 2 - (cx // 2)
    starty = y // 2 - (cy // 2)
    return img[starty:starty + cy, startx:startx + cx]


def resize(img, new_size, preserve_range=True):
    return transform.resize(img, new_size, preserve_range=preserve_range)

