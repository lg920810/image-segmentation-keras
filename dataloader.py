from tensorflow import keras
import math
import glob
import os
from augmentation import *

class DataGenerator(keras.utils.Sequence):
    '''data folder
       train
       --- image
          --- 1.jpg
          --- 2.jpg
          --- ...
       ---mask
          --- 1.bmp
          --- 2.bmp
          --- ...
        val
       --- image
          --- 1.jpg
          --- 2.jpg
          --- ...
       ---mask
          --- 1.bmp
          --- 2.bmp
          --- ...
    '''

    def __init__(self, path='', batch_size=1, resize=(256, 256), num_classes=3, shuffle=True):
        self.batch_size = batch_size
        self.path = path
        self.resize = resize
        self.num_classes = num_classes
        self.image_list = glob.glob(os.path.join(path, 'image') + '/*.jpg')
        self.indexes = np.arange(len(self.image_list))
        self.shuffle = shuffle

    def __len__(self):
        # 计算每一个epoch的迭代次数
        return math.ceil(len(self.image_list) / float(self.batch_size))

    def __getitem__(self, index):
        # 生成每个batch数据，这里就根据自己对数据的读取方式进行发挥了
        # 生成batch_size个索引
        batch_indexs = self.indexes[index * self.batch_size:(index + 1) * self.batch_size]
        # 根据索引获取datas集合中的数据
        batch_datas = [self.image_list[k] for k in batch_indexs]

        # 生成数据
        X, y = self.data_generation(batch_datas)

        return X, y

    def on_epoch_end(self):
        # 在每一次epoch结束是否需要进行一次随机，重新随机一下index
        if self.shuffle == True:
            np.random.shuffle(self.indexes)

    def data_generation(self, batch_datas):
        images = []
        labels = []

        # 生成数据
        for i, image_path in enumerate(batch_datas):
            # x_train数据
            image = cv2.imread(image_path)

            img = cv2.resize(image, self.resize)

            label_path = image_path[:-3] + 'bmp'
            label_path = label_path.replace('image', 'mask')

            if os.path.exists(label_path):
                mask = cv2.imread(label_path, cv2.IMREAD_GRAYSCALE)
                mask = cv2.resize(mask, self.resize)
                # print("found mask")
            else:
                mask = np.zeros(self.resize, dtype=np.int8)
                print("no mask")

            img = randomBrightness(img, gamma=0.5)
            img = randomBrightness(img, gamma=2)
            img = randomContrast(img, gain=1)
            img = randomMotionBlur(img)
            img = randomHueSaturationValue(img,
                                           hue_shift_limit=(-5, 5),
                                           sat_shift_limit=(-5, 5),
                                           val_shift_limit=(-5, 5))

            # if depth == 4:
            #     gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            #     b, g, r = cv2.split(img)
            #     img = cv2.merge([b, g, r, gray])

            all_masks = self.generate_all_masks(mask)
            images.append(img)
            labels.append(all_masks)

        images = np.array(images, np.float32) / 255.
        labels = np.array(labels, np.uint8)
        return np.array(images), np.array(labels)

    def generate_all_masks(self, mask):
        all_masks = np.zeros((self.resize[0], self.resize[1], self.num_classes))
        all_masks[mask == 255, 0] = 1
        all_masks[mask == 128, 1] = 1
        all_masks[mask == 0, 2] = 1
        return all_masks
