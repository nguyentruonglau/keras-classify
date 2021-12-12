from keras.preprocessing.image import load_img, img_to_array, array_to_img
from keras.preprocessing.image import smart_resize, save_img
import matplotlib.pyplot as plt
from tensorflow import keras
import matplotlib.cm as cm
import tensorflow as tf
import numpy as np
import os
import json


def get_img_array(img_path, size):
    """Load image from image path

    Args:
        img_path (string): path to image
        size (tuple): HxW
        
    Returns:
        [4D array]: image data
    """
    img = load_img(img_path)
    img = img_to_array(img)
    img = smart_resize(img, size)
    img = np.expand_dims(img, axis=0)
    return img


def make_gradcam_heatmap(img_array, model, index_last_activation_layer, pred_index=None):
    #Get last activate layer
    grad_model = tf.keras.models.Model(
        [model.inputs], [model.layers[index_last_activation_layer].output, model.output]
    )

    # Compute the gradient of the top predicted class for our input image
    # with respect to the activations of the last conv layer
    with tf.GradientTape() as tape:
        last_conv_layer_output, preds = grad_model(img_array)
        if pred_index is None:
            pred_index = tf.argmax(preds[0])
        class_channel = preds[:, pred_index]

    # This is the gradient of the output neuron (top predicted or chosen)
    # with regard to the output feature map of the last conv layer
    grads = tape.gradient(class_channel, last_conv_layer_output)

    # This is a vector where each entry is the mean intensity of the gradient
    # over a specific feature map channel
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

    # We multiply each channel in the feature map array
    # by "how important this channel is" with regard to the top predicted class
    # then sum all the channels to obtain the heatmap class activation
    last_conv_layer_output = last_conv_layer_output[0]
    heatmap = last_conv_layer_output @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)

    # For visualization purpose, we will also normalize the heatmap between 0 & 1
    heatmap = tf.maximum(heatmap, 0) / tf.math.reduce_max(heatmap)
    return heatmap.numpy()


def save_gradcam(img_path, heatmap, cam_path, input_shape, alpha=0.4):
    # Load the original image
    img = load_img(img_path)
    img = img_to_array(img)
    img = smart_resize(img, input_shape)

    # Rescale heatmap to a range 0-255
    heatmap = np.uint8(255 * heatmap)

    # Use jet colormap to colorize heatmap
    jet = cm.get_cmap("jet")

    # Use RGB values of the colormap
    jet_colors = jet(np.arange(256))[:, :3]
    jet_heatmap = jet_colors[heatmap]

    # Create an image with RGB colorized heatmap
    jet_heatmap = array_to_img(jet_heatmap)
    jet_heatmap = jet_heatmap.resize((img.shape[1], img.shape[0]))
    jet_heatmap = keras.preprocessing.image.img_to_array(jet_heatmap)

    # Superimpose the heatmap on original image
    superimposed_img = jet_heatmap * alpha + img
    superimposed_img = array_to_img(superimposed_img)

    # Save the superimposed image
    superimposed_img.save(cam_path)


def load_pretrained_model(path_to_pretrain):
    """Load pretrained model from path to file h5.

    Args:
        path_to_pretrain (string): path to pretrained model
         
    Returns:
        [model object] Model
    """
    model = None

    if os.path.exists(path_to_pretrain):
        if path_to_pretrain.endswith('.h5'):
            model = tf.keras.models.load_model(path_to_pretrain)
        else:
            raise ValueError('[Error]: H5 file is required')
    else:
        raise ValueError('[Error]: {} is not exists'.format(path_to_pretrain))

    return model


def get_target_names(json_label_decode):
    """Get encode of label

    Args: 
        json_label_decode (string): path to json file

    Returns:
        [dict] encode of label
    """
    if json_label_decode:
        with open(json_label_decode) as f:
            label_decode = json.load(f)

        target_names = []
        for i in label_decode:
                target_names.append(label_decode[i])

        return target_names
        
    else: raise ValueError('[ERROR]: {} is not found'.format(json_label_decode))