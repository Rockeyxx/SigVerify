"""Blue-ink signature cropping for scanned or photographed paper boxes."""
import cv2
import numpy as np

from config.settings import (INK_BACKGROUND_KERNEL, INK_CLUSTER_GAP, INK_CROP_PAD,
                             INK_HUE_RANGE, INK_LINE_MAX_BLUE, INK_LINE_MIN_FRACTION,
                             INK_MIN_CONTRAST, INK_MIN_PIXELS, INK_MIN_SATURATION)


def _ink_contrast(image):
    """Return local paper brightness minus pixel brightness."""
    grey = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    paper = cv2.medianBlur(grey, INK_BACKGROUND_KERNEL)
    return paper.astype(np.int16) - grey.astype(np.int16)


def _is_blue(image):
    """Return mask of blue-ink pixels, excluding grey shadows, creases, and black print."""
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    return ((hsv[:, :, 1] >= INK_MIN_SATURATION)
            & (hsv[:, :, 0] >= INK_HUE_RANGE[0]) & (hsv[:, :, 0] <= INK_HUE_RANGE[1]))


def _binarize(flat):
    return cv2.threshold(cv2.GaussianBlur(flat, (3, 3), 0), 0, 255,
                         cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]


def _printed_lines(binary, blue):
    """Mask long horizontal or vertical runs that are mostly non-blue (printed borders)."""
    ink = (binary == 0).astype(np.uint8)
    h, w = ink.shape
    runs = np.zeros_like(ink)
    for size in [(max(3, int(INK_LINE_MIN_FRACTION * w)), 1), (1, max(3, int(INK_LINE_MIN_FRACTION * h)))]:
        runs |= cv2.morphologyEx(ink, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_RECT, size))
    runs = cv2.dilate(runs, np.ones((3, 3), np.uint8)) & ink
    count, labels = cv2.connectedComponents(runs, connectivity=8)
    blue_share = (np.bincount(labels.ravel(), weights=blue.ravel(), minlength=count)
                  / np.maximum(np.bincount(labels.ravel(), minlength=count), 1))
    printed = blue_share < INK_LINE_MAX_BLUE
    printed[0] = False
    return printed[labels]


def ink_box(image):
    """Return padded (top, bottom, left, right) of the main blue-ink cluster, or None."""
    ink = ((_ink_contrast(image) >= INK_MIN_CONTRAST) & _is_blue(image)).astype(np.uint8)
    near = cv2.dilate(ink, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (INK_CLUSTER_GAP,) * 2))
    count, labels = cv2.connectedComponents(near, connectivity=8)
    if count < 2:
        return None
    sizes = np.bincount(labels[ink > 0], minlength=count)
    sizes[0] = 0
    signature = (labels == sizes.argmax()) & (ink > 0)
    if signature.sum() < INK_MIN_PIXELS:
        return None
    ys, xs = np.nonzero(signature)
    pad = int(INK_CROP_PAD * max(ys.max() - ys.min(), xs.max() - xs.min()))
    h, w = ink.shape
    return (max(0, ys.min() - pad), min(h, ys.max() + pad + 1),
            max(0, xs.min() - pad), min(w, xs.max() + pad + 1))


def ink_crop(image):
    """Crop the signature to flattened greyscale (no binarization; model expects grey scans), or None."""
    box = ink_box(image)
    if box is None:
        return None
    top, bottom, left, right = box
    flat = (255 - np.clip(_ink_contrast(image), 0, 255)).astype(np.uint8)[top:bottom, left:right]
    flat[_printed_lines(_binarize(flat), _is_blue(image)[top:bottom, left:right])] = 255
    return flat
