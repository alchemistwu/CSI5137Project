from model_utils import *
import matplotlib.pyplot as plt
import numpy as np
from tensorflow.keras.callbacks import CSVLogger, EarlyStopping
import os
import shutil
import argparse

class SaveBestModel(tf.keras.callbacks.Callback):
    """
    Callbacks for saving the model with lowest val_acc
    """
    def __init__(self, filepath, model_name, monitor='val_acc'):
        super(SaveBestModel, self).__init__()
        self.model_name = model_name
        self.best_weights = None
        self.file_path = filepath
        self.best = -float('inf')
        self.monitor = monitor

    def on_epoch_end(self, epoch, logs=None):
        current = logs.get(self.monitor)
        if not current:
            current = -float('inf')

        if np.less(self.best, current):
            self.best = current
            self.best_weights = self.model.get_weights()

            if not os.path.exists(self.file_path):
                os.mkdir(self.file_path)

            new_path = os.path.join(self.file_path, self.model_name)
            if not os.path.exists(new_path):
                os.mkdir(new_path)
            else:
                shutil.rmtree(new_path)

            new_path_model = os.path.join(new_path, 'model.tf')
            self.model.save_weights(new_path_model)
            print("New best model has been saved in %s!" % new_path_model)
            print("Best acc: %.4f" % current)

def get_tfDataset(training_directory, test_directory, verbose=False, batch_size=32):
    """
    loading directories into tensorflow dataset format
    :param training_directory:
    :param test_directory:
    :param verbose: if Ture, showing some sample from both training and testing dataset.
    :param batch_size:
    :return: tensorflow dataset object
    """
    dataTrain = tf.keras.preprocessing.image_dataset_from_directory(
    training_directory,
    labels="inferred",
    label_mode="int",
    color_mode="rgb",
    batch_size=batch_size,
    image_size=(256, 256),
    shuffle=True,
    seed=1,
    validation_split=None,
    subset=None,
    interpolation="bilinear",
    follow_links=False,
    )
    dataTrain = dataTrain.map(lambda x, y: (x / 255., y))
    dataTest = tf.keras.preprocessing.image_dataset_from_directory(
        test_directory,
        labels="inferred",
        label_mode="int",
        color_mode="rgb",
        batch_size=batch_size,
        image_size=(256, 256),
        shuffle=False,
        seed=1,
        validation_split=None,
        subset=None,
        interpolation="bilinear",
        follow_links=False,
    )

    dataTest = dataTest.map(lambda x, y: (x / 255., y))
    if verbose:
        plt.figure(figsize=(10, 10))
        for i, (images, labels) in enumerate(dataTrain.take(1)):
            for j in range(9):
                ax = plt.subplot(3, 3, j + 1)
                image = images[j, :, :, :]
                label = labels[j]
                plt.imshow(image, cmap='gray', vmin=0., vmax=1.)
                plt.title(int(label))
                plt.axis("off")
        plt.show()

        plt.figure(figsize=(10, 10))
        for i, (images, labels) in enumerate(dataTest.take(1)):
            for j in range(9):
                ax = plt.subplot(3, 3, j + 1)
                image = images[j, :, :, :]
                label = labels[j]
                plt.imshow(image, cmap='gray', vmin=0., vmax=1.)
                plt.title(int(label))
                plt.axis("off")
        plt.show()
    return dataTrain, dataTest

def train(train_directory, test_directory, model_name='res', epoch=100, batch_size=64, multi_gpu=True, pretrain=False,
          row=False):
    if not os.path.exists('logs'):
        os.mkdir('logs')
    if row:
        train_directory += "row"
        test_directory += "row"
    dataTrain, dataTest = get_tfDataset(train_directory, test_directory, batch_size=batch_size)
    dataTrain = dataTrain.cache()
    dataTrain = dataTrain.prefetch(tf.data.experimental.AUTOTUNE)
    dataTest = dataTest.cache()
    dataTest = dataTest.prefetch(tf.data.experimental.AUTOTUNE)

    if multi_gpu:
        strategy = tf.distribute.MirroredStrategy()
    else:
        strategy = tf.distribute.OneDeviceStrategy(device="/gpu:0")
    with strategy.scope():
        model = get_model(name=model_name, pretrain=True, target_size=256, n_class=9)
        if row:
            model_name += "_row"
        if pretrain:
            model_name += "_pretrain"
        model.fit(dataTrain, epochs=epoch, validation_data=dataTest, verbose=1, batch_size=batch_size,
                  callbacks=[SaveBestModel('model', model_name), CSVLogger('logs/%s.csv' % (model_name)),
                             EarlyStopping(monitor='val_loss', patience=10)])


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('-single_gpu', type=bool,
                        help='Training with single GPU support, default is multiple gpu',
                        dest='multi_gpu', const=False, default=True, nargs='?')

    parser.add_argument('-pretrain', type=bool,
                        help='Initialize the model with ImageNet pretrained weights',
                        dest='pretrain', const=True, default=False, nargs='?')

    parser.add_argument('-row', type=bool,
                        help='use the row wise entropy comparison',
                        dest='row', const=True, default=False, nargs='?')

    parser.add_argument('-train_dir', type=str,
                        help='Training directory',
                        dest='train_dir', default="../data/train_imgs")

    parser.add_argument('-test_dir', type=str,
                        help='Test directory',
                        dest='test_dir',
                        default="../data/test_imgs")

    parser.add_argument('-batch_size', type=int,
                        help='batch size',
                        dest='batch_size',
                        default=32)

    parser.add_argument('-epoch', type=int,
                        help='number of epochs',
                        dest='epoch',
                        default=100)

    parser.add_argument('-model', type=str,
                        help=''' Models to be used, options are: 'res','vgg','googLeNet','dense','mobile', defaut is 'res' ''',
                        dest='model',
                        default='res')
    args = parser.parse_args()
    train(args.train_dir, args.test_dir, multi_gpu=args.multi_gpu,
          batch_size=args.batch_size, epoch=args.epoch, model_name=args.model, pretrain=args.pretrain, row=args.row)
