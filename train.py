# -*- coding:utf-8 -*-
import os

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import tensorflow as tf

config = tf.ConfigProto()
config.gpu_options.allow_growth = True
session = tf.Session(config=config)
import argparse
from keras.utils import multi_gpu_model
import keras.optimizers as optim
from keras.callbacks import ReduceLROnPlateau, ModelCheckpoint, TensorBoard, CSVLogger
from dataloader import DataGenerator

from losses import dice_coeff, bce_dice_loss, keras_lovasz_hinge
from models.deeplabv3 import Deeplabv3
from models.unet import UNet

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("model", type=str, choices=['unet', 'deeplabv3+', 'fpn-vnet'],
                        help="Select the model to use training")
    parser.add_argument("--image_size", type=int, default=256,
                        help="Size of image")
    parser.add_argument("--image_depth", type=int, default=3,
                        help="Depth of image")
    parser.add_argument("--num_classes", type=int, default=3,
                        help="The number of classes to be segmented")
    parser.add_argument("--batch_size", type=int, default=4,
                        help="Number of examples per batch")
    parser.add_argument("--lr", type=float, default=1e-3,
                        help="learning rate for the optimizer")
    parser.add_argument("--epochs", type=int, default=50,
                        help="The number of epochs to train for")
    parser.add_argument("--gpus", type=int, default=1,
                        help="GPUs ID")
    parser.add_argument("--dataset", type=str, default=os.curdir,
                        help="Path to segmentation dataset")
    parser.add_argument("--outputs", type=str, default='outputs',
                        help="Training output location")
    parser.add_argument("--mode", type=str, default='train', choices=['train', 'validation'],
                        help="Train or verification")
    parser.add_argument("--weights", type=str, default=None,
                        help="The path of weights to be loaded")
    parser.add_argument("--loss", type=str, default='bce_dice_loss', choices=['bce_dice_loss', 'dice_loss', 'bce_loss'],
                        help="The loss function for traing")
    parser.add_argument("--comparison", type=bool, default=False,
                        help="comparison of integers of different signs")

    args = parser.parse_args()
    args.outputs = args.outputs + '_' + args.model
    if not os.path.exists(args.outputs):
        os.mkdir(args.outputs)

    # instantiate model
    if args.model == 'deeplabv3+':
        model = Deeplabv3(input_shape=[args.image_size, args.image_size, args.image_depth],
                          classes=args.num_classes,
                          backbone='mobilenetv2',
                          OS=16,
                          alpha=1.)
    elif args.model == 'unet':
        model = UNet(input_size=[args.image_size, args.image_size, args.image_depth],
                     classes=args.num_classes)
    else:
        pass

    model.summary()

    if (args.gpus > 1):
        model = multi_gpu_model(model, gpus=args.gpus)
    # train_mode = False
    if args.mode == 'validation':
        print("inference")
        model.load_weights(filepath="outputs_4_channels/best_model_final.h5", by_name=True)
        model.evaluate()
    else:
        if args.weights is not None:
            model.load_weights(filepath=args.weigths,
                               by_name=True,
                               skip_mismatch=True)
        optimizer = optim.adam(lr=args.lr, decay=5e-4)

        model.compile(optimizer=optimizer,
                      loss=bce_dice_loss,
                      metrics=[dice_coeff])

        log_filename = os.path.join(args.outputs, 'logger_model_train.csv')
        csv_log = CSVLogger(log_filename, separator=',', append=True)

        lr = ReduceLROnPlateau(monitor='val_loss',
                               factor=0.8,
                               patience=5,
                               verbose=2,
                               mode='min')

        ckpt_path = os.path.join(args.outputs, "best_weight_model_{epoch:03d}_{val_loss:.4f}.h5")
        ckpt = ModelCheckpoint(monitor='val_loss',
                               filepath=ckpt_path,
                               save_best_only=True,
                               save_weights_only=True,
                               verbose=1,
                               mode='min')

        tensorboard = TensorBoard(log_dir=args.outputs,
                                  histogram_freq=0,
                                  write_graph=True,
                                  write_images=True)

        callbacks = [lr, ckpt, tensorboard]

        train_generator = DataGenerator(path=os.path.join(args.dataset, 'train'),
                                        batch_size=args.batch_size)

        valid_generator = DataGenerator(path=os.path.join(args.dataset, 'val'),
                                        batch_size=args.batch_size)

        print('Training on {} samples'.format(len(train_generator)))
        print('Validating on {} samples'.format(len(valid_generator)))

        model.fit_generator(generator=train_generator,
                            initial_epoch=0,
                            epochs=args.epochs,
                            verbose=1,
                            validation_data=valid_generator,
                            callbacks=callbacks,
                            max_queue_size=20,
                            )

        model.save(os.path.join(args.outputs, 'best_model_final.h5'))
