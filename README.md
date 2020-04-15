# image-segmentation-keras
keras实现使用Deeplab v3+ UNet DANet网络的语义分割，可进行多模型对比

### 安装需要的包

例如 pip install keras(cv2,argparse,PIL,skimage,numpy,matplotlib,os,pandas等)


### 损失函数
在model文件夹中，包括三个基础分割网络

——Deeplab v3+

——UNet

——DANEt


### 损失函数和度量指标

在losses.py文件中，包含
iou loss，softmax dice loss，weighted loss，jaccard，f1score，focal_loss
dice for binary，weighted_bce_dice_loss，BINARY LOSSES，MULTICLASS LOSSES
precision，recall


### 训练 
```
python train.py deeplabv3+ --image_size 2 --batch_size 2
```
### 测试
```
python inference.py
```
