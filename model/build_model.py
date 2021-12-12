from keras.optimizers import Adam, Adadelta
from keras.callbacks import ModelCheckpoint
from keras import Model

from model.models import load_model, load_pretrained_model
import tensorflow as tf

from utils import rand_augment
from utils import ensure_dir

import multiprocessing
import warnings
import keras
import os

import pandas as pd
import numpy as np


class Classifier(object):
    def __init__(self,
                 model_name=None,
                 output_dir = None,
                 epochs = None,
                 batch_size = None,
                 optimizer = None,
                 learning_rate = None,
                 input_shape = None,
                 n_class = None,
                 use_pretrain = None,
                 path_to_pretrain = None,
                 use_augment = None,
                 monitor_to = None,
                 num_workers = None
        ):
        self.model_name = model_name
        self.output_dir = output_dir
        self.epochs = epochs
        self.batch_size = batch_size
        self.optimizer = optimizer
        self.input_shape = input_shape
        self.learning_rate = learning_rate
        self.n_class = n_class
        self.cnn_model = None
        self.use_pretrain=use_pretrain
        self.path_to_pretrain=path_to_pretrain
        self.use_augment=use_augment
        self.monitor_to=monitor_to
        self.num_workers=num_workers


    def _get_model(self):
        '''Get model architecture | pretrained model from path
        '''
        if self.use_pretrain:
            model = load_pretrained_model(self.path_to_pretrain)
        else:
            model = load_model(self.model_name, self.input_shape, self.n_class)
        return model

    def _get_metrics(self):
        '''Metric is a function that is used to judge the performance of your model.
        '''
        if self.n_class > 2:
            metrics = [
                      keras.metrics.CategoricalAccuracy(name='categorical_accuracy'),
                      keras.metrics.AUC(name='auc')
                ]
        elif self.n_class == 2:
            metrics = [
                      keras.metrics.BinaryAccuracy(name='binary_accuracy'),
                      keras.metrics.AUC(name='auc')
                ]
        return metrics

    def _get_losses(self):
        '''Losses
        '''
        if self.n_class>2: return tf.keras.losses.CategoricalCrossentropy(label_smoothing=0., from_logits=False)
        elif self.n_class==2: return tf.keras.losses.BinaryCrossentropy(label_smoothing=0., from_logits=False)

    def _get_lrschedule(self, epoch, lr):
        '''Learning rate scheduler
        '''
        warmup_lr = 1e-6
        init_lr = self.learning_rate
        warmup_epochs = 10

        if epoch < warmup_epochs:
            current_epoch_lr = warmup_lr + epoch * (init_lr - warmup_lr)/warmup_epochs
        else:
            current_epoch_lr = init_lr * (
                    1.0 + tf.math.cos(np.pi / (self.epochs-warmup_epochs)*(epoch-warmup_epochs))) / 2.0
        return current_epoch_lr

    def _get_optimizer(self):
        '''We provide two optimization algorithms: adadelta | adam
        '''
        opt = self.optimizer.lower()
        if opt == "adadelta":
            print("[INFOR]: Get optimizer->Adadelta")
            return Adadelta(
                learning_rate=1e-6, 
                rho=0.95,
                epsilon=1e-07,
                )
        print("[INFOR]: Get optimizer->Adam")
        return Adam(
                learning_rate=1e-6, 
                beta_1=0.9, 
                beta_2=0.999, 
                epsilon=1e-07,
                amsgrad=False
                )

    def _get_train_callbacks(self):
        '''Save model according callback list
        '''
        if self.n_class > 2:
            if self.monitor_to in ['val_categorical_accuracy', 'val_auc']:
                cp_best_auc = self._get_callbacks_save(monitor=self.monitor_to, mode="max")
                clr = self._get_callbacks_lrs()
                return [cp_best_auc, clr]
            else: raise ValueError('[ERROR]: Monitor not found')

        elif self.n_class == 2:
            if self.monitor_to in ['val_binary_accuracy', 'val_auc']:
                cp_best_auc = self._get_callbacks_save(monitor=self.monitor_to, mode="max")
                clr = self._get_callbacks_lrs()
                return [cp_best_auc, clr]
            else: raise ValueError('[ERROR]: Monitor not found')
               
    def _get_callbacks_lrs(self):
        '''Change learning rate, local minimum restrictions
        '''
        return tf.keras.callbacks.LearningRateScheduler(self._get_lrschedule)

    def _get_callbacks_save(self, monitor, mode):
        '''Save best model according monitor
        '''
        ensure_dir(self.output_dir)
        filepath = os.path.join(self.output_dir,  monitor + "_" + mode + ".h5")
        return ModelCheckpoint(filepath, monitor = monitor, mode = mode, save_best_only = True)

    def _save_history(self):
        '''Save model training history
        '''
        print("[INFOR]: Save->History")
        hist_df = pd.DataFrame(self._history.history)
        hf = os.path.join(self.output_dir + "/history.json")
        ensure_dir(self.output_dir)
        with open(hf, mode='w') as f: hist_df.to_json(f)

    def _init_model(self):
        '''Model initialization
        '''
        print("[INFOR]: Init_model->model_name: ", self.model_name)

        model = self._get_model()
        optimizer = self._get_optimizer()
        metrics = self._get_metrics()
        losses = self._get_losses()

        model.compile(
            loss=losses,
            optimizer=optimizer,
            metrics=metrics
        )
        self.cnn_model = model

        print("[INFOR]: Init_model->end")

    def _get_cpu_core(self):
        '''Check the validity of num_workers
        '''
        num_workers = self.num_workers
        cpu_core = multiprocessing.cpu_count()//2

        if num_workers == -1: num_workers=cpu_core
        if (num_workers < 0 and num_workers != -1) or num_workers > cpu_core:
            warnings.warn('[WARNING]: num_workers do not match, used num_workers=1 as default')
            num_workers=1

        return num_workers

    def _train(self, x_train, y_train, x_val, y_val):
        print("[INFOR]: Train->Start")

        ls_cback = self._get_train_callbacks()
        epochs = self.epochs
        num_workers = self._get_cpu_core()
        
        if self.use_augment:
            #for train data
            train_ds_rand = (
                tf.data.Dataset.from_tensor_slices((x_train, y_train))
                .shuffle(self.batch_size * 100)
                .batch(self.batch_size)
                # The returned output of `tf.py_function` contains an unncessary axis of
                # 1-D and we need to remove it.
                .map(
                    lambda x, y: (tf.py_function(rand_augment, [x], [tf.float32])[0], y),
                    num_parallel_calls=tf.data.AUTOTUNE,
                )
                .prefetch(tf.data.AUTOTUNE)
            )
            #for test data
            test_ds = (
                tf.data.Dataset.from_tensor_slices((x_val, y_val))
                .batch(self.batch_size)
                .prefetch(tf.data.AUTOTUNE)
            )

            self._history = self.cnn_model.fit(
                train_ds_rand, 
                validation_data=test_ds,
                epochs = epochs, 
                verbose = 1,
                callbacks=ls_cback,
                workers=num_workers,
                use_multiprocessing=True
                )
        else:
            #for train data
            train_ds_rand = (
                tf.data.Dataset.from_tensor_slices((x_train, y_train))
                .shuffle(self.batch_size * 100)
                .batch(self.batch_size)
                .prefetch(tf.data.AUTOTUNE)
            )
            #for test data
            test_ds = (
                tf.data.Dataset.from_tensor_slices((x_val, y_val))
                .batch(self.batch_size)
                .prefetch(tf.data.AUTOTUNE)
            )
            #for save history
            self._history = self.cnn_model.fit(
                train_ds_rand, 
                validation_data=test_ds,
                epochs = epochs, 
                verbose = 1,
                callbacks=ls_cback,
                workers=num_workers,
                use_multiprocessing=True
                )
            
        print("[INFOR]: Train->End")