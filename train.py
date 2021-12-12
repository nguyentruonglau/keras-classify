from model.build_model import Classifier
from keras.utils.layer_utils import count_params
import tensorflow as tf
from utils import get_data
import argparse
import numpy as np
import warnings
import os


#Compiled using XLA, auto-clustering on GPU
os.environ['TF_XLA_FLAGS'] = '--tf_xla_enable_xla_devices'
os.environ['TF_XLA_FLAGS'] = '--tf_xla_auto_jit=2'

#re-writes the environment variables and makes only certain NVIDIA GPU(s) visible for that process.
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]="0" #one GPU used, have id=0

#dynamic allocate GPU memory
physical_devices = tf.config.list_physical_devices('GPU')
if physical_devices: tf.config.experimental.set_memory_growth(physical_devices[0], True)
else: 
    warnings.warn('[WARNING]: GPU not found, CPU current is being used')


def main(FLAGS):
    print("[INFOR]: START TRAINING\n\n")
    
    input_dir = FLAGS.input_dir
    output_dir = FLAGS.output_dir
    model_name = FLAGS.model_name
    epochs = FLAGS.epochs
    batch_size = FLAGS.batch_size
    optimizer = FLAGS.optimizer
    learning_rate = FLAGS.learning_rate
    input_shape = FLAGS.input_shape
    use_pretrain = FLAGS.use_pretrain
    path_to_pretrain = FLAGS.path_to_pretrain
    n_class = FLAGS.n_class
    use_augment = FLAGS.use_augment
    monitor_to = FLAGS.monitor_to
    num_workers = FLAGS.num_workers
    
    cls = Classifier(
            output_dir = output_dir,
            model_name = model_name,
            epochs = epochs,
            batch_size = batch_size,
            optimizer = optimizer,
            learning_rate = learning_rate,
            input_shape = input_shape,
            n_class = n_class,
            use_pretrain = use_pretrain,
            path_to_pretrain = path_to_pretrain,
            use_augment = use_augment,
            monitor_to = monitor_to,
            num_workers = num_workers
            )

    cls._init_model()

    print('\n[INFOR]: Number of trainable parameter: {}'.format(count_params(cls.cnn_model.trainable_weights)))
    print('\n[INFOR]: Number of non-trainable parameter: {}'.format(count_params(cls.cnn_model.non_trainable_weights)))
    
    #get data
    x_train = get_data(input_dir, "x_train")
    y_train = get_data(input_dir, "y_train")
    
    x_val = get_data(input_dir, "x_val")
    y_val = get_data(input_dir, "y_val")

    #training
    cls._train(x_train, y_train, x_val, y_val)
    cls._save_history() 

    print("[INFOR]: END TRAINING DEEP CNN\n\n")


if __name__ == "__main__":
    FLAGS = None
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--input_dir',
        type=str,
        default='./convert/dataset',
        help='Input data directory'
    )
    parser.add_argument(
        '--output_dir',
        type=str,
        default='database/models/saved',
        help='Output directory of model'
    )
    parser.add_argument(
        '--model_name',
        type=str,
        default='EfficientNetB0',
        help=
        '''Build model at model/models
        '''
    )
    parser.add_argument(
        '--n_class',
        type=int,
        default=2,
        help='Number of classes'
    )
    parser.add_argument(
        '--epochs',
        type=int,
        default=15,
        help='Number of epochs'
    )
    parser.add_argument(
        '--batch_size',
        type=int,
        default=4,
        help='Please input batch_size'
    )
    parser.add_argument(
        '--optimizer',
        type=str,
        default='adam',
        help='Optimizer: adam, adadelta'
    )
    parser.add_argument(
        '--learning_rate',
        type=float,
        default=1e-3,
        help='Learning rate'
    )
    parser.add_argument(
        '--input_shape',
        type=tuple,
        default=(200, 260, 3), #HxWxC
        help='Input image shape'
    )
    parser.add_argument(
        '--use_pretrain',
        type=bool,
        default=False,
        help='Use pretrained model'
    )
    parser.add_argument(
        '--path_to_pretrain',
        type=str,
        default=None,
        help=
        '''Path to pretrained model folder
        '''
    )
    parser.add_argument(
        '--use_augment',
        type=bool,
        default=True,
        help=
        '''Augmentation on train data
        '''
    )
    parser.add_argument(
        '--monitor_to',
        type=str,
        default='val_binary_accuracy',
        help=
        '''You can monitor: 
        val_binary_accuracy, val_categorical_accuracy, val_auc
        '''
    )
    parser.add_argument(
        '--num_workers',
        type=int,
        default=4,
        help=
        '''Maximum number of processes to spin up 
           when using process-based threading.
        '''
    )
    FLAGS = parser.parse_args()
    print ("input_dir=",FLAGS.input_dir)
    print ("output_dir=",FLAGS.output_dir)
    print ("model_name=",FLAGS.model_name)
    print ("epochs=",FLAGS.epochs)
    print ("batch_size=",FLAGS.batch_size)
    print ("optimizer=",FLAGS.optimizer)
    print ("learning_rate=",FLAGS.learning_rate)
    print ("use_pretrain=",FLAGS.use_pretrain)
    print ("path_to_pretrain=",FLAGS.path_to_pretrain)
    print ("input_shape=",FLAGS.input_shape)
    print ("use_augment=",FLAGS.use_augment)
    print ("monitor_to=",FLAGS.monitor_to)
    print ("num_workers=",FLAGS.num_workers)

    main(FLAGS)