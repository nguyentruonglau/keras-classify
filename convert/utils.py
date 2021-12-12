from sklearn.model_selection import train_test_split
from imutils.paths import list_images

from keras.preprocessing.image import load_img, smart_resize
from keras.preprocessing.image import img_to_array
from keras.preprocessing.image import array_to_img
from keras.preprocessing.image import save_img

import tensorflow as tf
from tensorflow.keras.utils import to_categorical

from tqdm import tqdm
import numpy as np
import argparse
import warnings
import json
import os


def save_data(output_dir, fname, data):
    """Save data to npy file

    Args:
        output_dir (string): path to output folder
        fname (string): name npy file
        data (array): data to save
    Returns:
        None
    """
    f = os.path.join(output_dir, fname + ".npy")
    ensure_dir(output_dir)
    np.save(f, data)


def ensure_dir(directory):
    """Make sure the directory exists

    Args:
        directory (string): name of directory
         
    Returns:
        None
    """
    if not os.path.exists(directory):
        warnings.warn('''[Warning]: Output directory not found.
            The default output directory will be created.''')
        os.makedirs(directory)


def convert_to_npy(input_dir, output_dir, input_shape):
    """Save data training to npy file

    Args:
        input_dir (string): path to data folder
            input_dir -- class_one_dir
                            --image_1.jpg
                            --image_2.png
                      -- class_two_dir
                            --image_1.jpeg
                            --image_2.png
                      -- ..
        output_dir (string): path to output folder
    Returns:
        None
    """
    ls_dir_names = os.listdir(input_dir)
    num_classes = len(ls_dir_names)

    if len(ls_dir_names) == 0:
        raise ValueError("[Error]: There is no data directory in the {} directory.".format(input_dir))

    x_data = []
    y_data = []
    label_decode = dict()

    for i, dir_name in enumerate(ls_dir_names):
        path_to_images = os.path.join(input_dir, dir_name)
        image_paths = list(list_images(path_to_images))

        # add label
        label_decode[i] = dir_name

        # read images
        print("[INFOR]: Load images from {} directory.".format(dir_name))
        for j in tqdm(range(len(image_paths))):
            img = load_img(image_paths[j]) #RGB
            img = img_to_array(img)
            img = smart_resize(img, input_shape) #HxW
            x_data.append(img)
            y_data.append(float(i))

    x_train, x_test, y_train, y_test = train_test_split(x_data, y_data, 
                                                        test_size=0.1, 
                                                        random_state=42,
                                                        shuffle=True
                                                    )

    x_test, x_val, y_test, y_val = train_test_split(x_test, y_test,
                                                        test_size=0.5,
                                                        random_state=42,
                                                        shuffle=True
                                                    )
    #convert list to array
    x_train = np.array(x_train)
    x_val = np.array(x_val)
    x_test = np.array(x_test)

    #convert to onehot vector
    y_train = to_categorical(y=y_train, num_classes=num_classes)
    y_test = to_categorical(y=y_test, num_classes=num_classes)
    y_val = to_categorical(y=y_val, num_classes=num_classes)


    print('x_train shape = ', x_train.shape)
    print('y_train shape = ', y_train.shape)
    print('x_val shape = ', x_val.shape)
    print('y_val shape = ', y_val.shape)
    print('x_test shape = ', x_test.shape)
    print('y_test shape = ', y_test.shape)

    #ensure output_dir
    ensure_dir(output_dir)

    # save data to npy file
    save_data(output_dir, "x_train", x_train)
    save_data(output_dir, "x_test", x_test)
    save_data(output_dir, "x_val", x_val)

    save_data(output_dir, "y_train", y_train)
    save_data(output_dir, "y_test", y_test)
    save_data(output_dir, "y_val", y_val)


    # json file: dog -> 0, cat -> 1,... label decode corresponding
    with open(os.path.join(output_dir, 'label_decode.json'), 'w') as f:
            json.dump(label_decode, f, indent=4)


def convert_to_npy_custom(input_dir, output_dir, input_shape):
    """Save data training-validation-test to npy file

    Args:
        input_dir (string): path to data folder
            input_dir -- train
                            --class_one
                                --image_1.jpg
                                --image_2.png
                            --class_two
                                --image_1.jpeg
                                --image_2.png
                            -- ..
                      -- val
                            --class_one
                                --image_1.jpg
                                --image_2.png
                            --class_two
                                --image_1.jpeg
                                --image_2.png
                            -- ..
                      -- test
                            --class_one
                                --image_1.jpg
                                --image_2.png
                            --class_two
                                --image_1.jpeg
                                --image_2.png
                            -- ..
        output_dir (string): path to output folder
    Returns:
        None
    """
    ls_dir_names_outside = os.listdir(input_dir)

    if len(ls_dir_names_outside) == 0:
        raise ValueError("[Error]: There is no data directory in the {} directory.".format(input_dir))
    elif len(ls_dir_names_outside) != 3:
        raise ValueError("[Error]: In order to excatly evaluate your model,\
            data requirement needs to be divided into 3 sets: training set, validation set and test set.")

    num_classes = 0
    label_decode = dict()

    for i, dir_name_outside in enumerate(ls_dir_names_outside):
        x_data = []
        y_data = []

        ls_dir_names_inside = os.listdir(os.path.join(input_dir, dir_name_outside))
        num_classes = len(ls_dir_names_inside)

        print("[INFOR]: Go to {} directory.".format(dir_name_outside))
        for j, dir_name_inside in enumerate(ls_dir_names_inside):

            path_to_images = os.path.join(input_dir, dir_name_outside, dir_name_inside)
            image_paths = list(list_images(path_to_images))

            # add label
            label_decode[j] = dir_name_inside

            # read images
            print("[INFOR]: Load images from {} directory.".format(dir_name_inside))
            for k in tqdm(range(len(image_paths))):
                img = load_img(image_paths[k]) #RGB
                img = img_to_array(img)
                img = smart_resize(img, input_shape) #HxW -> WxH
                x_data.append(img)
                y_data.append(float(j))

    
        #shuffle data
        ls_permutation = np.random.permutation(len(x_data))
        n = len(ls_permutation)

        x = [x_data[ls_permutation[iter_]] for iter_ in range(n)]
        y = [y_data[ls_permutation[iter_]] for iter_ in range(n)]

        #convert x_data to array, convert y_data to onehot vector
        x_data = np.array(x)
        y_data = np.array(y)

        y_data = to_categorical(y=y_data, num_classes=num_classes)        

        print('x_{} shape = '.format(dir_name_outside), x_data.shape)
        print('y_{} shape = '.format(dir_name_outside), y_data.shape)

        #ensure output_dir
        ensure_dir(output_dir)

        # save data to npy file
        save_data(output_dir, "x_{}".format(dir_name_outside), x_data)
        save_data(output_dir, "y_{}".format(dir_name_outside), y_data)

        del x_data
        del y_data

    # json file: dog -> 0, cat -> 1,... label decode corresponding
    with open(os.path.join(output_dir, 'label_decode.json'), 'w') as f:
            json.dump(label_decode, f, indent=4)