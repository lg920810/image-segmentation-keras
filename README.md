#####image-segmentation-keras


###
安装需要的包
例如 pip install keras(cv2,argparse,PIL,skimage,numpy,matplotlib,os,pandas等)


###
语义分割模型
在model文件夹中
——deeplab v3
——unet

###
损失函数和度量指标
在losses.py文件中
包含了iou loss，softmax dice loss，weighted loss，jaccard，f1score，focal_loss
dice for binary，weighted_bce_dice_loss，BINARY LOSSES，MULTICLASS LOSSES
precision，recall


###
训练
运行 python train.py (若要修改参数，在train.py修改即可)
或不在train.py修改参数，运行例如python train.py --image_size --batch_size

###
测试
运行python inference.py
