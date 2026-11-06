"""CIFAR-10-C corruptions for 32x32 RGB images.

Ported from `ImageNet-C/create_c/make_cifar_c.py` in
https://github.com/hendrycks/robustness (Apache License 2.0), the code that made
CIFAR-10-C (Hendrycks and Dietterich, "Benchmarking Neural Network Robustness to Common
Corruptions and Perturbations", ICLR 2019). The CIFAR parameters are unchanged. Changes:
`multichannel=True` is now `channel_axis=-1` (scikit-image), `np.float_` is
`np.float64` (NumPy 2), and `random_noise` takes a seed so runs repeat. Motion blur,
snow and frost are left out (they need ImageMagick or extra image files), and so is
glass blur (slow).

Each function takes a PIL image and a severity from 1 to 5, and returns an array or a
PIL image with values in [0, 255]. Randomness uses NumPy's global generator, as in the
original; seed it with `np.random.seed`.
"""

from io import BytesIO

import cv2
import numpy as np
import skimage as sk
from PIL import Image as PILImage
from scipy.ndimage import map_coordinates
from scipy.ndimage import zoom as scizoom
from skimage.filters import gaussian


def disk(radius, alias_blur=0.1, dtype=np.float32):
    if radius <= 8:
        L = np.arange(-8, 8 + 1)
        ksize = (3, 3)
    else:
        L = np.arange(-radius, radius + 1)
        ksize = (5, 5)
    X, Y = np.meshgrid(L, L)
    aliased_disk = np.array((X**2 + Y**2) <= radius**2, dtype=dtype)
    aliased_disk /= np.sum(aliased_disk)
    # supersample disk to antialias
    return cv2.GaussianBlur(aliased_disk, ksize=ksize, sigmaX=alias_blur)


# modification of https://github.com/FLHerne/mapgen/blob/master/diamondsquare.py
def plasma_fractal(mapsize=32, wibbledecay=3):
    """Heightmap from the diamond-square algorithm, values in [0, 1]."""
    assert mapsize & (mapsize - 1) == 0
    maparray = np.empty((mapsize, mapsize), dtype=np.float64)
    maparray[0, 0] = 0
    stepsize = mapsize
    wibble = 100

    def wibbledmean(array):
        return array / 4 + wibble * np.random.uniform(-wibble, wibble, array.shape)

    def fillsquares():
        cornerref = maparray[0:mapsize:stepsize, 0:mapsize:stepsize]
        squareaccum = cornerref + np.roll(cornerref, shift=-1, axis=0)
        squareaccum += np.roll(squareaccum, shift=-1, axis=1)
        maparray[stepsize // 2 : mapsize : stepsize, stepsize // 2 : mapsize : stepsize] = wibbledmean(squareaccum)

    def filldiamonds():
        mapsize_ = maparray.shape[0]
        drgrid = maparray[stepsize // 2 : mapsize_ : stepsize, stepsize // 2 : mapsize_ : stepsize]
        ulgrid = maparray[0:mapsize_:stepsize, 0:mapsize_:stepsize]
        ldrsum = drgrid + np.roll(drgrid, 1, axis=0)
        lulsum = ulgrid + np.roll(ulgrid, -1, axis=1)
        ltsum = ldrsum + lulsum
        maparray[0:mapsize_:stepsize, stepsize // 2 : mapsize_ : stepsize] = wibbledmean(ltsum)
        tdrsum = drgrid + np.roll(drgrid, 1, axis=1)
        tulsum = ulgrid + np.roll(ulgrid, -1, axis=0)
        ttsum = tdrsum + tulsum
        maparray[stepsize // 2 : mapsize_ : stepsize, 0:mapsize_:stepsize] = wibbledmean(ttsum)

    while stepsize >= 2:
        fillsquares()
        filldiamonds()
        stepsize //= 2
        wibble /= wibbledecay

    maparray -= maparray.min()
    return maparray / maparray.max()


def clipped_zoom(img, zoom_factor):
    h = img.shape[0]
    ch = int(np.ceil(h / zoom_factor))
    top = (h - ch) // 2
    img = scizoom(img[top : top + ch, top : top + ch], (zoom_factor, zoom_factor, 1), order=1)
    trim_top = (img.shape[0] - h) // 2
    return img[trim_top : trim_top + h, trim_top : trim_top + h]


def gaussian_noise(x, severity=1):
    c = [0.04, 0.06, 0.08, 0.09, 0.10][severity - 1]
    x = np.array(x) / 255.0
    return np.clip(x + np.random.normal(size=x.shape, scale=c), 0, 1) * 255


def shot_noise(x, severity=1):
    c = [500, 250, 100, 75, 50][severity - 1]
    x = np.array(x) / 255.0
    return np.clip(np.random.poisson(x * c) / c, 0, 1) * 255


def impulse_noise(x, severity=1):
    c = [0.01, 0.02, 0.03, 0.05, 0.07][severity - 1]
    x = sk.util.random_noise(np.array(x) / 255.0, mode="s&p", amount=c, rng=np.random.randint(2**31))
    return np.clip(x, 0, 1) * 255


def speckle_noise(x, severity=1):
    c = [0.06, 0.1, 0.12, 0.16, 0.2][severity - 1]
    x = np.array(x) / 255.0
    return np.clip(x + x * np.random.normal(size=x.shape, scale=c), 0, 1) * 255


def gaussian_blur(x, severity=1):
    c = [0.4, 0.6, 0.7, 0.8, 1][severity - 1]
    x = gaussian(np.array(x) / 255.0, sigma=c, channel_axis=-1)
    return np.clip(x, 0, 1) * 255


def defocus_blur(x, severity=1):
    c = [(0.3, 0.4), (0.4, 0.5), (0.5, 0.6), (1, 0.2), (1.5, 0.1)][severity - 1]
    x = np.array(x) / 255.0
    kernel = disk(radius=c[0], alias_blur=c[1])
    channels = [cv2.filter2D(x[:, :, d], -1, kernel) for d in range(3)]
    channels = np.array(channels).transpose((1, 2, 0))
    return np.clip(channels, 0, 1) * 255


def zoom_blur(x, severity=1):
    c = [np.arange(1, 1.06, 0.01), np.arange(1, 1.11, 0.01), np.arange(1, 1.16, 0.01),
         np.arange(1, 1.21, 0.01), np.arange(1, 1.26, 0.01)][severity - 1]
    x = (np.array(x) / 255.0).astype(np.float32)
    out = np.zeros_like(x)
    for zoom_factor in c:
        out += clipped_zoom(x, zoom_factor)
    x = (x + out) / (len(c) + 1)
    return np.clip(x, 0, 1) * 255


def fog(x, severity=1):
    c = [(0.2, 3), (0.5, 3), (0.75, 2.5), (1, 2), (1.5, 1.75)][severity - 1]
    x = np.array(x) / 255.0
    max_val = x.max()
    x += c[0] * plasma_fractal(wibbledecay=c[1])[:32, :32][..., np.newaxis]
    return np.clip(x * max_val / (max_val + c[0]), 0, 1) * 255


def spatter(x, severity=1):
    c = [(0.62, 0.1, 0.7, 0.7, 0.5, 0), (0.65, 0.1, 0.8, 0.7, 0.5, 0), (0.65, 0.3, 1, 0.69, 0.5, 0),
         (0.65, 0.1, 0.7, 0.69, 0.6, 1), (0.65, 0.1, 0.5, 0.68, 0.6, 1)][severity - 1]
    x = np.array(x, dtype=np.float32) / 255.0
    liquid_layer = np.random.normal(size=x.shape[:2], loc=c[0], scale=c[1])
    liquid_layer = gaussian(liquid_layer, sigma=c[2])
    liquid_layer[liquid_layer < c[3]] = 0
    if c[5] == 0:
        liquid_layer = (liquid_layer * 255).astype(np.uint8)
        dist = 255 - cv2.Canny(liquid_layer, 50, 150)
        dist = cv2.distanceTransform(dist, cv2.DIST_L2, 5)
        _, dist = cv2.threshold(dist, 20, 20, cv2.THRESH_TRUNC)
        dist = cv2.blur(dist, (3, 3)).astype(np.uint8)
        dist = cv2.equalizeHist(dist)
        ker = np.array([[-2, -1, 0], [-1, 1, 1], [0, 1, 2]])
        dist = cv2.filter2D(dist, cv2.CV_8U, ker)
        dist = cv2.blur(dist, (3, 3)).astype(np.float32)
        m = cv2.cvtColor(liquid_layer * dist, cv2.COLOR_GRAY2BGRA)
        m /= np.max(m, axis=(0, 1))
        m *= c[4]
        # water is pale turquoise
        color = np.concatenate((175 / 255.0 * np.ones_like(m[..., :1]), 238 / 255.0 * np.ones_like(m[..., :1]),
                                238 / 255.0 * np.ones_like(m[..., :1])), axis=2)
        color = cv2.cvtColor(color, cv2.COLOR_BGR2BGRA)
        x = cv2.cvtColor(x, cv2.COLOR_BGR2BGRA)
        return cv2.cvtColor(np.clip(x + m * color, 0, 1), cv2.COLOR_BGRA2BGR) * 255
    m = np.where(liquid_layer > c[3], 1, 0)
    m = gaussian(m.astype(np.float32), sigma=c[4])
    m[m < 0.8] = 0
    # mud brown
    color = np.concatenate((63 / 255.0 * np.ones_like(x[..., :1]), 42 / 255.0 * np.ones_like(x[..., :1]),
                            20 / 255.0 * np.ones_like(x[..., :1])), axis=2)
    color *= m[..., np.newaxis]
    x *= 1 - m[..., np.newaxis]
    return np.clip(x + color, 0, 1) * 255


def contrast(x, severity=1):
    c = [0.75, 0.5, 0.4, 0.3, 0.15][severity - 1]
    x = np.array(x) / 255.0
    means = np.mean(x, axis=(0, 1), keepdims=True)
    return np.clip((x - means) * c + means, 0, 1) * 255


def brightness(x, severity=1):
    c = [0.05, 0.1, 0.15, 0.2, 0.3][severity - 1]
    x = np.array(x) / 255.0
    x = sk.color.rgb2hsv(x)
    x[:, :, 2] = np.clip(x[:, :, 2] + c, 0, 1)
    x = sk.color.hsv2rgb(x)
    return np.clip(x, 0, 1) * 255


def saturate(x, severity=1):
    c = [(0.3, 0), (0.1, 0), (1.5, 0), (2, 0.1), (2.5, 0.2)][severity - 1]
    x = np.array(x) / 255.0
    x = sk.color.rgb2hsv(x)
    x[:, :, 1] = np.clip(x[:, :, 1] * c[0] + c[1], 0, 1)
    x = sk.color.hsv2rgb(x)
    return np.clip(x, 0, 1) * 255


def jpeg_compression(x, severity=1):
    c = [80, 65, 58, 50, 40][severity - 1]
    output = BytesIO()
    x.save(output, "JPEG", quality=c)
    return PILImage.open(output)


def pixelate(x, severity=1):
    c = [0.95, 0.9, 0.85, 0.75, 0.65][severity - 1]
    x = x.resize((int(32 * c), int(32 * c)), PILImage.BOX)
    return x.resize((32, 32), PILImage.BOX)


# mod of https://gist.github.com/erniejunior/601cdf56d2b424757de5
def elastic_transform(image, severity=1):
    IMSIZE = 32
    c = [(IMSIZE * 0, IMSIZE * 0, IMSIZE * 0.08), (IMSIZE * 0.05, IMSIZE * 0.2, IMSIZE * 0.07),
         (IMSIZE * 0.08, IMSIZE * 0.06, IMSIZE * 0.06), (IMSIZE * 0.1, IMSIZE * 0.04, IMSIZE * 0.05),
         (IMSIZE * 0.1, IMSIZE * 0.03, IMSIZE * 0.03)][severity - 1]
    image = np.array(image, dtype=np.float32) / 255.0
    shape = image.shape
    shape_size = shape[:2]
    # random affine
    center_square = np.float32(shape_size) // 2
    square_size = min(shape_size) // 3
    pts1 = np.float32([center_square + square_size, [center_square[0] + square_size, center_square[1] - square_size],
                       center_square - square_size])
    pts2 = pts1 + np.random.uniform(-c[2], c[2], size=pts1.shape).astype(np.float32)
    M = cv2.getAffineTransform(pts1, pts2)
    image = cv2.warpAffine(image, M, shape_size[::-1], borderMode=cv2.BORDER_REFLECT_101)
    dx = (gaussian(np.random.uniform(-1, 1, size=shape[:2]), c[1], mode="reflect", truncate=3) * c[0]).astype(np.float32)
    dy = (gaussian(np.random.uniform(-1, 1, size=shape[:2]), c[1], mode="reflect", truncate=3) * c[0]).astype(np.float32)
    dx, dy = dx[..., np.newaxis], dy[..., np.newaxis]
    x, y, z = np.meshgrid(np.arange(shape[1]), np.arange(shape[0]), np.arange(shape[2]))
    indices = np.reshape(y + dy, (-1, 1)), np.reshape(x + dx, (-1, 1)), np.reshape(z, (-1, 1))
    return np.clip(map_coordinates(image, indices, order=1, mode="reflect").reshape(shape), 0, 1) * 255


CORRUPTIONS = {
    f.__name__: f
    for f in [gaussian_noise, shot_noise, impulse_noise, speckle_noise, gaussian_blur, defocus_blur,
              zoom_blur, fog, spatter, contrast, brightness, saturate, jpeg_compression, pixelate,
              elastic_transform]
}
"""Supported corruptions, named as the CIFAR-10-C files."""


def corrupt(img_hwc_uint8: np.ndarray, name: str, severity: int) -> np.ndarray:
    """Corrupt one HxWx3 uint8 image; returns HxWx3 uint8, as CIFAR-10-C stores it."""
    out = CORRUPTIONS[name](PILImage.fromarray(img_hwc_uint8), severity)
    return np.uint8(out)
