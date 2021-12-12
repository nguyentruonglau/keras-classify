from tensorflow.keras import layers
import tensorflow as tf
import numpy as np
import os


def load_model(model_name, input_shape, n_class):
    """Get model from model name. You can adjust according to the requirements of the program such as:
    initializing weights='imagenet' or include_top=True.

    Args:
        model_name (string): name of model
        input_shape (tuple): input shape of images
        n_class (int): number of class
         
    Returns:
        [model object] Model
    """
    input_tensor = tf.keras.Input(shape=input_shape)
    scaled_images = tf.keras.layers.experimental.preprocessing.Rescaling(scale=1./127.5, offset=-1)(input_tensor)

    # See more model at: https://www.tensorflow.org/api_docs/python/tf/keras/applications
    base_model = tf.keras.applications.EfficientNetB0(weights='imagenet', input_shape=input_shape, include_top=False)
    # base_model.trainable = False

    x = base_model(scaled_images, training=False)

    # Convert features of shape `base_model.output_shape[1:]` to vectors
    x = tf.keras.layers.GlobalAveragePooling2D()(x)

    # A Dense classifier with a single unit (binary classifications)
    outputs = tf.keras.layers.Dense(units=n_class, activation='softmax')(x)
    model = tf.keras.Model(input_tensor, outputs)

    model.summary()
    return model


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
        raise ValueError('[Error]: {} is not exists')

    return model