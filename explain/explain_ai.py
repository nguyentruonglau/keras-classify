from utils import load_pretrained_model
from utils import make_gradcam_heatmap
from imutils.paths import list_images
from utils import get_target_names
from utils import get_img_array
from utils import save_gradcam
from tensorflow import keras
import tensorflow as tf
from tqdm import tqdm
import numpy as np
import argparse
import os

os.environ['TF_XLA_FLAGS'] = '--tf_xla_enable_xla_devices'

def main(FLAGS):
    #get parameters
    input_dir=FLAGS.input_dir
    output_dir=FLAGS.output_dir
    pretrained_model_path=FLAGS.pretrained_model_path
    input_shape=FLAGS.input_shape
    json_label_decode = FLAGS.json_label_decode
    index_last_activation_layer =FLAGS.index_last_activation_layer

    #get target names
    target_names = get_target_names(json_label_decode)

    #make model
    model = load_pretrained_model(pretrained_model_path)
    model.summary()
    print('[INFOR]: Pretrained have loaded')

    # Remove last layer's softmax
    model.layers[-1].activation = None

    #get image paths
    img_paths = list(list_images(input_dir))
    for i in tqdm(range(len(img_paths))):
        #prepare image
        img_array = get_img_array(img_paths[i], size=input_shape)
     
        # Print what the top predicted class is
        preds = model.predict(img_array)
        index = np.argmax(preds, axis=1)[0]
        
        # Generate class activation heatmap
        heatmap = make_gradcam_heatmap(img_array, model, index_last_activation_layer)
        bs_name = os.path.basename(img_paths[i])

        fname =  target_names[index] + '_' + bs_name

        cam_path = os.path.join(output_dir, fname)
        save_gradcam(img_paths[i], heatmap, cam_path, input_shape)

    print("[INFOR]: See result at: {} folder".format(output_dir))


if __name__ == '__main__':

    FLAGS = None
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--input_dir',
        type=str,
        default='input',
        help='Input data directory'
    )
    parser.add_argument(
        '--output_dir',
        type=str,
        default='output',
        help='Output data directory'
    )
    parser.add_argument(
        '--pretrained_model_path',
        type=str,
        default='../data/output/model/val_accuracy_max.h5',
        help=
        ''' Path to pretrained model
        '''
    )
    parser.add_argument(
        '--input_shape',
        type=tuple,
        default=(52, 102),
        help='Shape of input, HxW'
    )
    parser.add_argument(
        '--json_label_decode',
        type=str,
        default='../convert/data/label_decode.json',
        help='path to label decode json file'
    )
    parser.add_argument(
        '--index_last_activation_layer',
        type=int,
        default=-4,
        help=
        '''
        Model.summary() 
        will help you to know the index of last activation layer
        '''
    )
    FLAGS = parser.parse_args()
    print("input_dir = ", FLAGS.input_dir)
    print("output_dir = ", FLAGS.output_dir)
    print("pretrained_model_path = ", FLAGS.pretrained_model_path)
    print("input_shape = ", FLAGS.input_shape)
    print("json_label_decode = ", FLAGS.json_label_decode)
    print("index_last_activation_layer = ", FLAGS.index_last_activation_layer)
    main(FLAGS)