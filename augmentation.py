from scipy.ndimage.filters import gaussian_filter
from scipy.ndimage.interpolation import map_coordinates
from skimage import exposure
import cv2
import numpy as np


def color_normalization(img):
    img_normalized = np.empty(img.shape)
    for i in range(img.shape[2]):
        img_temp = img[:, :, i]
        img_temp_std = np.std(img_temp)
        img_temp_mean = np.mean(img_temp)
        img_temp_normalized = (img_temp - img_temp_mean) / img_temp_std
        img_normalized[:, :, i] = ((img_temp_normalized - np.min(img_temp_normalized)) / (
                np.max(img_temp_normalized) - np.min(img_temp_normalized))) * 255
        return img_normalized


def randomHueSaturationValue(image,
                             hue_shift_limit=(-180, 180),
                             sat_shift_limit=(-255, 255),
                             val_shift_limit=(-255, 255),
                             u=0.5):
    if np.random.random() < u:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        h, s, v = cv2.split(image)
        hue_shift = np.random.uniform(hue_shift_limit[0], hue_shift_limit[1])
        h = cv2.add(h, hue_shift)
        sat_shift = np.random.uniform(sat_shift_limit[0], sat_shift_limit[1])
        s = cv2.add(s, sat_shift)
        val_shift = np.random.uniform(val_shift_limit[0], val_shift_limit[1])
        v = cv2.add(v, val_shift)
        image = cv2.merge((h, s, v))
        image = cv2.cvtColor(image, cv2.COLOR_HSV2BGR)

    return image


def randomShiftScaleRotate(image, mask,
                           shift_limit=(-0.0625, 0.0625),
                           scale_limit=(-0.1, 0.1),
                           rotate_limit=(-45, 45),
                           aspect_limit=(0, 0),
                           border_mode=cv2.BORDER_CONSTANT,
                           u=0.5):
    if np.random.random() < u:
        height, width, channel = image.shape

        angle = np.random.uniform(rotate_limit[0], rotate_limit[1])  # degree
        scale = np.random.uniform(1 + scale_limit[0], 1 + scale_limit[1])
        aspect = np.random.uniform(1 + aspect_limit[0], 1 + aspect_limit[1])
        sx = scale * aspect / (aspect ** 0.5)
        sy = scale / (aspect ** 0.5)
        dx = round(np.random.uniform(shift_limit[0], shift_limit[1]) * width)
        dy = round(np.random.uniform(shift_limit[0], shift_limit[1]) * height)

        cc = np.math.cos(angle / 180 * np.math.pi) * sx
        ss = np.math.sin(angle / 180 * np.math.pi) * sy
        rotate_matrix = np.array([[cc, -ss], [ss, cc]])

        box0 = np.array([[0, 0], [width, 0], [width, height], [0, height], ])
        box1 = box0 - np.array([width / 2, height / 2])
        box1 = np.dot(box1, rotate_matrix.T) + np.array([width / 2 + dx, height / 2 + dy])

        box0 = box0.astype(np.float32)
        box1 = box1.astype(np.float32)
        mat = cv2.getPerspectiveTransform(box0, box1)
        image = cv2.warpPerspective(image, mat, (width, height),
                                    flags=cv2.INTER_LINEAR,
                                    borderMode=border_mode,
                                    borderValue=(0, 0, 0))
        mask = cv2.warpPerspective(mask, mat, (width, height),
                                   flags=cv2.INTER_NEAREST,
                                   borderMode=border_mode,
                                   borderValue=(0, 0, 0))
    return image, mask


def randomBrightness(image, gamma=0.5, u=0.5):
    if np.random.random() < u:
        image = exposure.adjust_gamma(image, gamma)

    return image


def randomContrast(image, gain=1, u=0.5):
    if np.random.random() < u:
        image = exposure.adjust_log(image, gain)
    return image


def randomHorizontalFlip(image, mask, u=0.5):
    if np.random.random() < u:
        image = cv2.flip(image, 1)
        mask = cv2.flip(mask, 1)

    return image, mask


def randomMotionBlur(image, degree=12, angle=45, u=0.5):
    if np.random.random() < u:
        M = cv2.getRotationMatrix2D((degree / 2, degree / 2), angle, 1)
        motion_blur_kernel = np.diag(np.ones(degree))
        motion_blur_kernel = cv2.warpAffine(motion_blur_kernel, M, (degree, degree))
        motion_blur_kernel = motion_blur_kernel / degree
        blurred = cv2.filter2D(image, -1, motion_blur_kernel)
        cv2.normalize(blurred, blurred, 0, 255, cv2.NORM_MINMAX)
        image = np.array(blurred, dtype=np.uint8)
    return image


def randomElasticTransformation(image, mask, alpha=1500, sigma=40, rng=np.random.RandomState(42),
                                interpolation_order=1, u=0.5):
    if np.random.random() < u:
        height, width, channel = image.shape
        # make random fields
        dx = rng.uniform(-1, 1, [height, width]) * alpha
        dy = rng.uniform(-1, 1, [height, width]) * alpha
        # smooth dx and dy
        sdx = gaussian_filter(dx, sigma=sigma, mode='reflect')
        sdy = gaussian_filter(dy, sigma=sigma, mode='reflect')
        # make meshgrid
        x, y = np.meshgrid(np.arange(width), np.arange(height))
        # distort meshgrid indices
        distorted_indices = (y + sdy).reshape(-1, 1), (x + sdx).reshape(-1, 1)
        # map coordinates from image to distorted index set
        all_channels = []
        for channel_id in range(channel):
            all_channels.append(map_coordinates(image[..., channel_id], distorted_indices, mode='reflect',
                                                order=interpolation_order).reshape(height, width, 1))
        transformed_image = np.concatenate(all_channels, axis=-1)
        transformed_mask = map_coordinates(mask, distorted_indices, mode='reflect',
                                           order=interpolation_order).reshape(height, width)
        return transformed_image, transformed_mask
    return image, mask
