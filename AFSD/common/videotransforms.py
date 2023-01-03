import numbers

import numpy as np
import vedacore.image as veda_image
from numpy import random
from PIL import Image


def imresize(img, size, interp="bicubic"):
    im = Image.fromarray(img)
    func = {"nearest": 0, "lanczos": 1, "bilinear": 2, "bicubic": 3, "cubic": 3}
    im = im.resize(size, func[interp])
    return np.array(im)


class ResizeClip(object):
    def __init__(self, size):
        if isinstance(size, numbers.Number):
            self.size = (int(size), int(size))
        else:
            self.size = size

    def __call__(self, imgs):
        imgs = np.transpose(imgs, [1, 2, 3, 0])
        res = []
        for i in range(imgs.shape[0]):
            res.append(imresize(imgs[i], self.size, "bicubic"))
        res = np.stack(res, 0)
        return res.transpose([3, 0, 1, 2])


class RandomCrop(object):
    """Crop the given video sequences (t x h x w) at a random location.
    Args:
        size (sequence or int): Desired output size of the crop. If size is an
            int instead of sequence like (h, w), a square crop (size, size) is
            made.
    """

    def __init__(self, size):
        if isinstance(size, numbers.Number):
            self.size = (int(size), int(size))
        else:
            self.size = size

    @staticmethod
    def get_params(img, output_size):
        """Get parameters for ``crop`` for a random crop.
        Args:
            img (PIL Image): Image to be cropped.
            output_size (tuple): Expected output size of the crop.
        Returns:
            tuple: params (i, j, h, w) to be passed to ``crop`` for random crop.
        """
        c, t, h, w = img.shape
        th, tw = output_size
        if w == tw and h == th:
            return 0, 0, h, w

        i = random.randint(0, h - th) if h != th else 0
        j = random.randint(0, w - tw) if w != tw else 0
        return i, j, th, tw

    def __call__(self, imgs):

        i, j, h, w = self.get_params(imgs, self.size)

        imgs = imgs[:, :, i : i + h, j : j + w]
        return imgs

    def __repr__(self):
        return self.__class__.__name__ + "(size={0})".format(self.size)


class CenterCrop(object):
    """Crops the given seq Images at the center.
    Args:
        size (sequence or int): Desired output size of the crop. If size is an
            int instead of sequence like (h, w), a square crop (size, size) is
            made.
    """

    def __init__(self, size):
        if isinstance(size, numbers.Number):
            self.size = (int(size), int(size))
        else:
            self.size = size

    def __call__(self, imgs):
        """
        Args:
            img (PIL Image): Image to be cropped.
        Returns:
            PIL Image: Cropped image.
        """
        c, t, h, w = imgs.shape
        th, tw = self.size
        i = int(np.round((h - th) / 2.0))
        j = int(np.round((w - tw) / 2.0))

        return imgs[:, :, i : i + th, j : j + tw]

    def __repr__(self):
        return self.__class__.__name__ + "(size={0})".format(self.size)


class RandomHorizontalFlip(object):
    """Horizontally flip the given seq Images randomly with a given probability.
    Args:
        p (float): probability of the image being flipped. Default value is 0.5
    """

    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, imgs):
        """
        Args:
            img (seq Images): seq Images to be flipped.
        Returns:
            seq Images: Randomly flipped seq images.
        """
        if random.random() < self.p:
            # c x t x h x w
            return np.flip(imgs, axis=3).copy()
        return imgs

    def __repr__(self):

        return self.__class__.__name__ + "(p={})".format(self.p)


class PhotoMetricDistortion(object):
    """Apply photometric distortion to images sequentially, every
    transformation is applied with a probability of 0.5. The position of random
    contrast is in second or second to last.

    1. random brightness
    2. random contrast (mode 0)
    3. convert color from BGR to HSV
    4. random saturation
    5. random hue
    6. convert color from HSV to BGR
    7. random contrast (mode 1)
    8. randomly swap channels

    Args:
        brightness_delta (int): delta of brightness.
        contrast_range (tuple): range of contrast.
        saturation_range (tuple): range of saturation.
        hue_delta (int): delta of hue.
    """

    def __init__(
        self,
        brightness_delta=32,
        contrast_range=(0.5, 1.5),
        saturation_range=(0.5, 1.5),
        hue_delta=18,
        p=0.5,
    ):
        self.brightness_delta = brightness_delta
        self.contrast_lower, self.contrast_upper = contrast_range
        self.saturation_lower, self.saturation_upper = saturation_range
        self.hue_delta = hue_delta
        self.p = p

    def __call__(self, imgs):
        """Call function to perform photometric distortion on images.

        Args:
            imgs (np.array): The input images. Shape: [C,T,H,W]

        Returns:
            dict: Result dict with images distorted.
        """

        assert imgs.dtype == np.float32, (
            "PhotoMetricDistortion needs the input imgs of dtype np.float32"
            ', please set "to_float32=True" in "LoadFrames" pipeline'
        )

        def _filter(img):
            img[img < 0] = 0
            img[img > 255] = 255
            return img

        if random.uniform(0, 1) <= self.p:
            imgs = np.transpose(imgs, (1, 2, 3, 0))  # [T,H,W,C]

            # random brightness
            if random.randint(2):
                delta = random.uniform(-self.brightness_delta, self.brightness_delta)
                imgs += delta
                imgs = _filter(imgs)

            # mode == 0 --> do random contrast first
            # mode == 1 --> do random contrast last
            mode = random.randint(2)
            if mode == 1:
                if random.randint(2):
                    alpha = random.uniform(self.contrast_lower, self.contrast_upper)
                    imgs *= alpha
                    imgs = _filter(imgs)

            # convert color from BGR to HSV
            imgs = np.array([veda_image.bgr2hsv(img) for img in imgs])

            # random saturation
            if random.randint(2):
                imgs[..., 1] *= random.uniform(self.saturation_lower, self.saturation_upper)

            # random hue
            # if random.randint(2):
            if True:
                imgs[..., 0] += random.uniform(-self.hue_delta, self.hue_delta)
                imgs[..., 0][imgs[..., 0] > 360] -= 360
                imgs[..., 0][imgs[..., 0] < 0] += 360

            # convert color from HSV to BGR
            imgs = np.array([veda_image.hsv2bgr(img) for img in imgs])
            imgs = _filter(imgs)

            # random contrast
            if mode == 0:
                if random.randint(2):
                    alpha = random.uniform(self.contrast_lower, self.contrast_upper)
                    imgs *= alpha
                    imgs = _filter(imgs)

            # randomly swap channels
            if random.randint(2):
                imgs = imgs[..., random.permutation(3)]

            imgs = imgs.transpose([3, 0, 1, 2])  # [C,T,H,W]
        return imgs


class Rotate(object):
    """Spatially rotate images.

    Args:
        limit (int, list or tuple): Angle range, (min_angle, max_angle).
        interpolation (str): Interpolation method, accepted values are
            "nearest", "bilinear", "bicubic", "area", "lanczos".
            Default: bilinear
        border_mode (str): Border mode, accepted values are "constant",
            "isolated", "reflect", "reflect101", "replicate", "transparent",
            "wrap". Default: constant
        border_value (int): Border value. Default: 0
    """

    def __init__(
        self,
        limit,
        interpolation="bilinear",
        border_mode="constant",
        border_value=0,
        p=0.5,
    ):
        if isinstance(limit, int):
            limit = (-limit, limit)
        self.limit = limit
        self.interpolation = interpolation
        self.border_mode = border_mode
        self.border_value = border_value
        self.p = p

    def __call__(self, imgs):
        """Call function to random rotate images.

        Args:
            imgs (np.array): The input images. Shape: [C,T,H,W]

        Returns:
            dict: Spatially rotated results.
        """

        if random.uniform(0, 1) <= self.p:
            imgs = np.transpose(imgs, [1, 2, 3, 0])  # [T,H,W,C]
            angle = random.uniform(*self.limit)
            imgs = [
                veda_image.imrotate(
                    img,
                    angle=angle,
                    interpolation=self.interpolation,
                    border_mode=self.border_mode,
                    border_value=self.border_value,
                )
                for img in imgs
            ]
            imgs = np.array(imgs)
            imgs = np.transpose(imgs, [3, 0, 1, 2])  # [C,T,H,W]
        return imgs

