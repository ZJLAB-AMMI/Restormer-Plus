import numpy as np
from PIL import Image, ImageChops, ImageOps, ImageEnhance


def sample_level(n):
    return np.random.uniform(low=0.1, high=n)


def int_parameter(level, maxval):
    """Helper function to scale `val` between 0 and maxval .

    Args:
      level: Level of the operation that will be between [0, `PARAMETER_MAX`].
      maxval: Maximum value that the operation can have. This will be scaled to
        level/PARAMETER_MAX.

    Returns:
      An int that results from scaling `maxval` according to `level`.
    """
    return int(level * maxval / 10)


def float_parameter(level, maxval):
    """Helper function to scale `val` between 0 and maxval.

    Args:
      level: Level of the operation that will be between [0, `PARAMETER_MAX`].
      maxval: Maximum value that the operation can have. This will be scaled to
        level/PARAMETER_MAX.

    Returns:
      A float that results from scaling `maxval` according to `level`.
    """
    return float(level) * maxval / 10.


def autocontrast(pil_img, _):
    return ImageOps.autocontrast(pil_img)


def equalize(pil_img, _):
    return ImageOps.equalize(pil_img)


def posterize(pil_img, level):
    level = int_parameter(sample_level(level), 4)
    return ImageOps.posterize(pil_img, 4 - level)


def rotate(pil_img, level):
    degrees = int_parameter(sample_level(level), 30)
    if np.random.uniform() > 0.5:
        degrees = -degrees
    return pil_img.rotate(degrees, resample=Image.BILINEAR)


def solarize(pil_img, level):
    level = int_parameter(sample_level(level), 256)
    return ImageOps.solarize(pil_img, 256 - level)


def shear_x(pil_img, level):
    level = float_parameter(sample_level(level), 0.3)
    if np.random.uniform() > 0.5:
        level = -level
    return pil_img.transform(
        (pil_img.width, pil_img.height),
        Image.AFFINE, (1, level, 0, 0, 1, 0),
        resample=Image.BILINEAR)


def shear_y(pil_img, level):
    level = float_parameter(sample_level(level), 0.3)
    if np.random.uniform() > 0.5:
        level = -level
    return pil_img.transform(
        (pil_img.width, pil_img.height),
        Image.AFFINE, (1, 0, 0, level, 1, 0),
        resample=Image.BILINEAR)


def roll_x(pil_img, level):
    """Roll an image sideways."""
    delta = int_parameter(sample_level(level), pil_img.width / 3)
    if np.random.random() > 0.5:
        delta = -delta
    xsize, ysize = pil_img.size
    delta = delta % xsize
    if delta == 0: return pil_img
    part1 = pil_img.crop((0, 0, delta, ysize))
    part2 = pil_img.crop((delta, 0, xsize, ysize))
    pil_img.paste(part1, (xsize - delta, 0, xsize, ysize))
    pil_img.paste(part2, (0, 0, xsize - delta, ysize))

    return pil_img


def roll_y(pil_img, level):
    """Roll an image sideways."""
    delta = int_parameter(sample_level(level), pil_img.width / 3)
    if np.random.random() > 0.5:
        delta = -delta
    xsize, ysize = pil_img.size
    delta = delta % ysize
    if delta == 0: return pil_img
    part1 = pil_img.crop((0, 0, xsize, delta))
    part2 = pil_img.crop((0, delta, xsize, ysize))
    pil_img.paste(part1, (0, ysize - delta, xsize, ysize))
    pil_img.paste(part2, (0, 0, xsize, ysize - delta))

    return pil_img


# operation that overlaps with ImageNet-C's test set
def color(pil_img, level):
    level = float_parameter(sample_level(level), 1.8) + 0.1
    return ImageEnhance.Color(pil_img).enhance(level)


# operation that overlaps with ImageNet-C's test set
def contrast(pil_img, level):
    level = float_parameter(sample_level(level), 1.8) + 0.1
    return ImageEnhance.Contrast(pil_img).enhance(level)


# operation that overlaps with ImageNet-C's test set
def brightness(pil_img, level):
    level = float_parameter(sample_level(level), 1.8) + 0.1
    return ImageEnhance.Brightness(pil_img).enhance(level)


# operation that overlaps with ImageNet-C's test set
def sharpness(pil_img, level):
    level = float_parameter(sample_level(level), 1.8) + 0.1
    return ImageEnhance.Sharpness(pil_img).enhance(level)


def zoom_x(pil_img, level):
    # zoom from .02 to 2.5
    rate = level
    zoom_img = pil_img.transform(
        (pil_img.width, pil_img.height),
        Image.AFFINE, (rate, 0, 0, 0, 1, 0),
        resample=Image.BILINEAR)
    # need to do reflect padding
    if rate > 1.0:
        orig_x, orig_y = pil_img.size
        new_x = int(orig_x / rate)
        zoom_img = np.array(zoom_img)
        zoom_img = np.pad(zoom_img[:, :new_x, :], ((0, 0), (0, orig_x - new_x), (0, 0)), 'wrap')
    return zoom_img


def zoom_y(pil_img, level):
    # zoom from .02 to 2.5
    rate = level
    zoom_img = pil_img.transform(
        (pil_img.width, pil_img.height),
        Image.AFFINE, (1, 0, 0, 0, rate, 0),
        resample=Image.BILINEAR)
    # need to do reflect padding
    if rate > 1.0:
        orig_x, orig_y = pil_img.size
        new_y = int(orig_y / rate)
        zoom_img = np.array(zoom_img)
        zoom_img = np.pad(zoom_img[:new_y, :, :], ((0, orig_y - new_y), (0, 0), (0, 0)), 'wrap')
    return zoom_img


augmentations = [
    rotate, shear_x, shear_y,
    zoom_x, zoom_y, roll_x, roll_y
]


