import cv2
import argparse
from keras.models import load_model
from PIL import Image
import numpy as np
from skimage import exposure, morphology
from models.deeplabv3 import Deeplabv3
import matplotlib.pyplot as plt
import os
import pandas as pd

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

seg_size = 256
depth = 4


def crop(image, center, radius=256):
    '''
    根据检测框坐标从原图裁剪目标区域，返回裁剪区域及新的裁剪中心

    :param image: 原图
    :param center: 通过检测框计算的裁剪中心
    :param radius: 圆半径也就是裁剪框边长的一半
    :return:裁剪区域及新的裁剪中心,该坐标用于映射回原图使用
    '''
    height, width = image.shape[1], image.shape[0]
    x, y = center

    if x - radius < 0:
        x = radius
    if x + radius > width:
        x = width - radius
    if y - radius < 0:
        y = radius
    if y + radius > height:
        y = height - radius

    if len(image.shape) >= 3:
        crop_image = image[y - radius:y + radius, x - radius:x + radius, :]
    else:
        crop_image = image[y - radius:y + radius, x - radius:x + radius]

    return crop_image, x, y


def predict_and_map2origin(image_path, ckpt_path, out_path):
    '''
    预测mask并将mask映射回原图大小
    :return:
    '''
    # model = load_model(ckpt_path, compile=False)
    model = Deeplabv3(input_shape=[seg_size, seg_size, depth],
                      classes=1,
                      backbone='mobilenetv2',
                      OS=16,
                      alpha=1.)
    model.load_weights(ckpt_path)
    model.summary()

    box_csv_path = os.path.join(image_path, 'box.csv')

    df = pd.read_csv(box_csv_path, encoding='utf-8')  # 读取mrnn输出的视盘检测的坐标以便裁剪

    for index in range(len(df)):
        image_name = df.iloc[index]['image']
        print(image_name)
        x = df.iloc[index]['x']
        y = df.iloc[index]['y']
        w = df.iloc[index]['w']
        h = df.iloc[index]['h']
        confidence = df.iloc[index]['confidence']

        if confidence < 0.8:  # 对于confidence小于0.8的就认为检测错了
            print(image_name, ' Disc position error.')
            continue
        if not os.path.exists(os.path.join(image_path, image_name)):
            continue
        image = cv2.imread(os.path.join(image_path, image_name))
        origin_h, origin_w = image.shape[0], image.shape[1]  # 得到图像宽，高

        center_x = (x + x + w) / 2
        center_y = (y + y + h) / 2

        radius = int(2 * (np.max([w, h]) // 2))  # 根据框的大小决定裁剪区域大小 2倍的检测框宽和高的最大值
        if radius < 300:
            radius = 400
        center = (int(center_x), int(center_y))
        crop_image, crop_x, crop_y = crop(image, center, radius=radius)  # 裁剪原图并返回实际裁剪的中心坐标

        # crop_image = cv2.cvtColor(crop_image, cv2.COLOR_BGR2RGB)
        crop_image = cv2.resize(crop_image, (seg_size, seg_size))
        ''' four channels'''
        if depth == 4:
            gray = cv2.cvtColor(crop_image, cv2.COLOR_BGR2GRAY)
            b, g, r = cv2.split(crop_image)
            crop_image = cv2.merge([b, g, r, gray])

        crop_image = crop_image.astype(np.float32) / 255.0

        crop_image = np.expand_dims(crop_image, axis=0)
        pred = model.predict(crop_image)

        output = pred.reshape(seg_size, seg_size)
        output[output >= 0.5] = 1
        output[output < 0.5] = 0
        # plt.imshow(output, cmap='gray')
        # plt.axis('off')
        # plt.show()
        # cv2.imwrite(os.path.join(out_path, image_name.split('.')[0] + '.bmp'), 255 * output)

        data = output.copy()
        data = data.astype(np.bool)
        data = morphology.remove_small_objects(data, min_size=100, connectivity=2, in_place=True)
        output = output * data

        output = cv2.resize(output, (2 * radius, 2 * radius), interpolation=cv2.INTER_NEAREST)

        output = cv2.medianBlur(output.astype(np.uint8), ksize=5)

        _, contours, hierarch = cv2.findContours(output, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_NONE)
        number = 10

        if len(contours) == 0 or len(contours) > number:
            print(image_name, len(contours))
            continue

        area_array = np.zeros([number, ])

        for i in range(len(contours)):
            area_array[i] = cv2.contourArea(contours[i])
        area_index = np.argsort(area_array)

        # 往空的mask上画，轮廓内填充值
        crop_mask = 255 * np.ones(shape=[2 * radius, 2 * radius], dtype=np.uint8)
        cv2.drawContours(crop_mask,
                         contours=contours,
                         contourIdx=area_index[number - 1],
                         color=(0, 0, 0),
                         thickness=-1)

        mapping = 255 * np.ones((origin_h, origin_w), dtype=np.uint8)
        mapping[crop_y - radius:crop_y + radius, crop_x - radius:crop_x + radius] = crop_mask

        plt.subplot(121)
        plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        plt.axis('off')
        plt.title(image_name)
        plt.subplot(122)
        plt.imshow(255 - mapping, cmap='gray')
        plt.axis('off')
        plt.show()

        # cv2.imwrite(os.path.join(out_path, image_name.split('.')[0] + '.bmp'), mapping)


if __name__ == '__main__':
    ckpt_path = 'opticDisc/weights/deeplabv3_opticdisc.h5'
    image_path = r'F:\PALM-final\PALM-Testing400-Images'
    out_path = r'F:\PALM-final\submit'

    predict_and_map2origin(image_path, ckpt_path, out_path)
