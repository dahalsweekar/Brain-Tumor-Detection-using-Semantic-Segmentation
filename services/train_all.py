import math

import tensorflow as tf
import os
import numpy as np
import gc

os.environ["SM_FRAMEWORK"] = "tf.keras"
from tensorflow import keras
import segmentation_models as sm
import tensorflow_advanced_segmentation_models as tasm
import sys

sys.path.append("/content/drive/MyDrive/Brain-Tumor-Detection-using-Semantic-Segmentation/")

from scripts.model import Models
from scripts.prepare_dataset import Prepare_Dataset
from scripts.visualizer import Visualizer
from scripts.score_all import Score
import warnings

warnings.filterwarnings('ignore')


def main():
    networks = ['unet', 'segnet', 'deeplabv3']
    backbones = ['vgg16', 'resnet50', 'densenet121', 'mobilenetv2', 'efficientnetb0']
    augment = [False, True]
    for nets in networks:
        for bb in backbones:
            for aug in augment:
                network = nets
                backbone = bb
                augm = aug
                print(f'\n__________________'
                      f'\nSetup: \nNetwork:{nets}\nBackbone:{bb}\nAugmentation:{augm}\n'
                      '__________________\n')
                Train(network=network, backbone=backbone, epoch=100, verbose=1, batch_size=8, validation_split=0.1,
                      test_split=0.2, weight_path='./models', visualizer=True, data_path='./data/BTD_Dataset'
                      , score=True, test=False, augment=augm, threshold=0.0305,
                      IMG_SIZE=256).train_model()
                gc.collect()


class Train:

    def __init__(self, network, backbone, epoch, verbose, batch_size, validation_split, test_split, weight_path,
                 visualizer, data_path, score, test, augment, threshold, IMG_SIZE=256):
        self.test_size = test_split
        self.network = network
        self.backbone = backbone
        self.augment = augment
        self.threshold = threshold
        self.IMG_SIZE = self.size_(IMG_SIZE)
        (self.Y_train_cat, self.Y_test_cat, self.X_train, self.Y_test, self.X_test, self.p_weights,
         self.n_classes) = Prepare_Dataset(self.IMG_SIZE, self.IMG_SIZE, self.augment, self.threshold,
                                           backbone=backbone,
                                           test_size=test_split,
                                           data_path=data_path).prepare_all()
        self.total_loss = Prepare_Dataset(self.IMG_SIZE, self.IMG_SIZE).get_loss(p_weights=self.p_weights)
        self.epoch = epoch
        self.verbose = verbose
        self.batch_size = batch_size
        self.validation_split = validation_split
        self.weight_path = weight_path
        self.visualizer = visualizer
        self.score = score
        self.test = test

    def validation_data_(self, fold):
        x_train_data = []
        y_train_data = []
        x_val_data = []
        y_val_data = []
        segments = math.ceil(0.2 * len(self.X_train))
        initial = fold * segments
        final = (fold + 1) * segments
        # X_validation
        for x_vals in self.X_train[initial:final]:
            x_val_data.append(x_vals)
        # Y_validation
        for y_vals in self.Y_train_cat[initial:final]:
            y_val_data.append(y_vals)
        # X_train
        for x_trains in self.X_train[0:initial]:
            x_train_data.append(x_trains)
        for x_trains in self.X_train[final:5 * segments]:
            x_train_data.append(x_trains)
        # Y_train
        for y_trains in self.Y_train_cat[0:initial]:
            y_train_data.append(y_trains)
        for y_trains in self.Y_train_cat[final:5 * segments]:
            y_train_data.append(y_trains)

        print(f'Training index: from 0 to {initial - 1} and from {final} to {5 * segments - 1}')
        print(f'Validation index: from {initial} to {final-1}')
        print(f'Training size: {len(x_train_data)}')
        print(f'Validation size: {len(x_val_data)}')

        x_val_data = np.array(x_val_data)
        y_val_data = np.array(y_val_data)
        x_train_data = np.array(x_train_data)
        y_train_data = np.array(y_train_data)

        return x_train_data, y_train_data, x_val_data, y_val_data

    def size_(self, IMG_SIZE):
        if self.network == 'pspnet':
            if IMG_SIZE % 48 != 0:
                print('Image size must be divisible by 48')
                IMG_SIZE = int(IMG_SIZE / 48) * 48 + 48
                print(f'New image size: {IMG_SIZE}x{IMG_SIZE}x3')
        return IMG_SIZE

    def train_model(self):
        print('Training...')

        metrics = [sm.metrics.IOUScore(threshold=0.5), sm.metrics.FScore(threshold=0.5)]
        LR = 0.0001
        optim = keras.optimizers.Adam(LR)

        # Initialize model
        if self.network == 'custom':
            model = Models(self.n_classes, self.IMG_SIZE, IMG_CHANNELS=3, model_name=self.network,
                           backbone=self.backbone).simple_unet_model()
        elif self.network == 'segnet':
            model, self.backbone = Models(self.n_classes, self.IMG_SIZE, IMG_CHANNELS=3, model_name=self.network,
                                          backbone=self.backbone).segnet_architecture()
        elif self.network == 'unet' or self.network == 'linknet' or self.network == 'pspnet':
            model = Models(self.n_classes, self.IMG_SIZE, IMG_CHANNELS=3, model_name=self.network,
                           backbone=self.backbone).segmented_models()
        elif self.network == 'deeplabv3':
            base_model, layers, layer_names = Models(self.n_classes, self.IMG_SIZE, IMG_CHANNELS=3,
                                                     model_name=self.network,
                                                     backbone=self.backbone).deeplabv3(name=self.backbone,
                                                                                       weights='imagenet',
                                                                                       height=self.IMG_SIZE,
                                                                                       width=self.IMG_SIZE)
            model = tasm.DeepLabV3plus(n_classes=self.n_classes, base_model=base_model, output_layers=layers,
                                       backbone_trainable=False)
            model.build((None, self.IMG_SIZE, self.IMG_SIZE, 3))
        else:
            print(f'{self.network} network is not available.')
            quit()
        # Compilation
        model.compile(optimizer=optim,
                      loss=self.total_loss,
                      metrics=[metrics])

        # Summary
        print(model.summary())

        # Callbacks
        weight_path = str(os.path.join(self.weight_path,
                                       self.network + '_' + self.backbone + '_' + str(
                                           self.IMG_SIZE) + '.hdf5'))
        checkpoint = tf.keras.callbacks.ModelCheckpoint(weight_path, verbose=self.verbose, save_best_only=True)
        callbacks = [
            tf.keras.callbacks.EarlyStopping(patience=15, monitor='val_loss')
        ]
        print('Applying 5-fold validation metrics')
        scores_ = []
        precision_ = []
        recall_ = []
        f1score_ = []
        means_ = []
        accuracy_ = []
        print(
            f'*********ARCHITECTURE*********** \n\tNetwork: {self.network}\n\tBackBone: {self.backbone}\n**************'
            f'******************')
        for fold in range(0, 5):
            x_train_data, y_train_data, x_val_data, y_val_data = self.validation_data_(fold=fold)
            print(f'________________________\nTraining with validation fold: {fold + 1}\n________________________')
            history = model.fit(x_train_data,
                                y_train_data,
                                batch_size=self.batch_size,
                                epochs=self.epoch,
                                verbose=self.verbose,
                                validation_data=(x_val_data, y_val_data),
                                callbacks=callbacks)
            # weight_path = weight_path[:-5] + f'_fold_{fold}.hdf5'
            # print(weight_path)
            Prepare_Dataset(self.IMG_SIZE, self.IMG_SIZE).save_model(model, path=weight_path)
            print(f'Model saved : {weight_path}')
            if self.visualizer:
                Visualizer(history,
                           name=f'_fold_{fold + 1}_{self.network}_{self.backbone}_augment_{self.augment}').plot_curve()
            if self.score:
                scrs, meanIOU, report = Score(model, weight_path, self.X_test, self.Y_test, self.n_classes,
                                      self.Y_test_cat).calc_scores()
                precision_.append(scrs['weighted avg']['precision'])
                recall_.append(scrs['weighted avg']['recall'])
                f1score_.append(scrs['weighted avg']['f1-score'])
                scores_.append(report)
                means_.append(meanIOU)
                accuracy_.append(scrs['accuracy'])
            if self.test:
                print('Argument test is not available for 5-fold cross validation')
            gc.collect()
        if self.score:
            k = 0
            f = open(f'./reports/Report_{self.network}_{self.backbone}_augment_{self.augment}.txt',
                     'w')
            for sc in scores_:
                k = k + 1
                # print(f'Report for fold {k}:' + str(sc) + '\n\n')
                w = f'_______________________________\nReport for fold {k}:\n_______________________________\n' \
                    + sc + '\n' + f'MeanIOU={means_[k - 1]}' + '\n\n'
                f.write(w)
            avg_precision = 0
            avg_recall = 0
            avg_f1 = 0
            avg_mean = 0
            avg_acc = 0
            for i in range(0, 5):
                avg_precision = avg_precision + float(precision_[i])
                avg_recall = avg_recall + float(recall_[i])
                avg_f1 = avg_f1 + float(f1score_[i])
                avg_mean = avg_mean + float(means_[i])
                avg_acc = avg_acc + float(accuracy_[i])
            a = '____________\nAverages\n____________\n' + f'Average Precision: {str(avg_precision / 5)}\n' \
                + f'Average Recall: {str(avg_recall / 5)}\n' + f'Average f1-score: {str(avg_f1 / 5)}\n' \
                + f'Average MeanIOU: {str(avg_mean / 5)}\n' + f'Average Accuracy: {str(avg_acc / 5)}\n'
            f.write(a)
            f.close()
            # FiveFoldVisualizer(scores=scores_,
            #                    id=f'{self.network}_{self.backbone}_augment_{self.augment}_binary_{self.binary}').plot_curve()


if __name__ == '__main__':
    main()
