import cv2
import numpy as np


def crop_center(img, crop_size):
    """Center crop images."""
    x, y = img.shape[:2]
    cx, cy = crop_size
    startx = x // 2 - (cx // 2)
    starty = y // 2 - (cy // 2)
    return img[starty:starty + cy, startx:startx + cx]


def resize(img, new_size):
    return cv2.resize(img, tuple(new_size[:2]))


def pad_square(img):
    im_shape = img.shape[:2]
    target_size = np.max(im_shape)
    h_pad = target_size - im_shape[0]
    w_pad = target_size - im_shape[1]
    t = h_pad // 2
    b = t
    l = w_pad // 2
    r = l
    return cv2.copyMakeBorder(img, t, b, l, r, cv2.BORDER_CONSTANT, 0.)
