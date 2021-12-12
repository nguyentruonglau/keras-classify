import os, sys; sys.path.append('..')

from tensorflow.keras.models import load_model
from utils import get_data, ensure_dir, evaluate
from model.models import load_pretrained_model
import numpy as np
import json
import argparse
import tensorflow as tf


#Compiled using XLA, auto-clustering on GPU
os.environ['TF_XLA_FLAGS'] = '--tf_xla_enable_xla_devices'
os.environ['TF_XLA_FLAGS'] = '--tf_xla_cpu_global_jit'

#re-writes the environment variables and makes only certain NVIDIA GPU(s) visible for that process.
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]="0" #one GPU used, have id=0

#dynamic allocate GPU memory
physical_devices = tf.config.list_physical_devices('GPU')
if physical_devices: tf.config.experimental.set_memory_growth(physical_devices[0], True)
else: 
    warnings.warn('[WARNING]: GPU not found, CPU current is being used')


def main(FLAGS):
    #get information
    input_dir = FLAGS.input_dir
    model_path = FLAGS.model_path
    json_label_decode = FLAGS.json_label_decode
    json_history = FLAGS.json_history
    model_name = FLAGS.model_name
    output_dir = FLAGS.output_dir

    #load data
    print("[INFOR]: Get test data.")
    x_test = get_data(input_dir, "x_test")
    y_test = get_data(input_dir, "y_test")

    #load pretrained model
    model = load_pretrained_model(model_path)
    print('[INFOR]: Pretrained model is loaded.')

    #predict
    y_score = model.predict(x_test)
    n_class = y_score.shape[1]

    ensure_dir(output_dir)

    #evaluate pretrained model
    evaluate(FLAGS, y_test, y_score, n_class)

    print()


if __name__ == '__main__':

    FLAGS = None
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--input_dir',
        type=str,
        default='../convert/dataset',
        help='input test data directory'
    )
    parser.add_argument(
        '--output_dir',
        type=str,
        default='../database/output',
        help='output directory'
    )
    parser.add_argument(
        '--model_path',
        type=str,
        default='../database/models/saved/val_binary_accuracy_max.h5',
        help='path to pretrained model'
    )
    parser.add_argument(
        '--model_name',
        type=str,
        default='EfficientNetB0',
        help=
        '''build model at model/models
        '''
    )
    parser.add_argument(
        '--json_label_decode',
        type=str,
        default='../convert/dataset/label_decode.json',
        help='path to label decode json file'
    )
    parser.add_argument(
        '--json_history',
        type=str,
        default='../database/models/saved/history.json',
        help=
        '''path to history json file
           if None, plot history is ignored
        '''
    )
    parser.add_argument(
        '--n_class',
        type=int,
        default=2,
        help='Number of classes'
    )
    FLAGS = parser.parse_args()
    print ("\n[INFOR]: input_dir=",FLAGS.input_dir)
    print ("[INFOR]: output_dir=",FLAGS.output_dir)
    print ("[INFOR]: model_path=",FLAGS.model_path)
    print ("[INFOR]: json_label_decode=",FLAGS.json_label_decode)
    print ("[INFOR]: json_history=", FLAGS.json_history)
    print ("[INFOR]: n_class=", FLAGS.n_class)
    print ("[INFOR]: model_name={}\n".format(FLAGS.model_name))
    main(FLAGS)