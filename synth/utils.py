from concurrent.futures import ThreadPoolExecutor
from imutils.paths import list_images
import imgaug.augmenters as iaa

import warnings
import multiprocessing
import time
import cv2
import os

import tensorflow as tf
import configparser as ConfigParser
from optparse import OptionParser
from numpy import expand_dims
from tqdm import tqdm
import numpy as np


def submit_thread(ls_img_paths, output_dir, num_imgs, num_workers):
    executor = ThreadPoolExecutor(max_workers=num_workers)
    execs = [executor.submit(generate_images, ls_img_paths, output_dir, num_imgs) for _ in range(num_workers)]


def ensure_dir(directory):
    """Make sure the directory exists
    """
    if not os.path.exists(directory):
        warnings.warn("[Warning] Output directory not found. \
            The default output directory will be created.")
        os.makedirs(directory)


def datagenerate():
    """Generating images in batches
    See more: https://github.com/aleju/imgaug
    """
    seq = iaa.Sequential([
        iaa.MultiplyHue((0.5, 1.5)),
        iaa.Fliplr(1.0),
        iaa.Affine(rotate=(-10, 10)),
        iaa.Affine(scale=(0.8, 1.2))
    ])
    return seq


def generate_images(ls_img_paths, output_dir, num_imgs):
    """Generate image with ImageDataGenerator
    """
    #Random select image to generate image
    for i in tqdm(range(num_imgs)):
        rd_image_seleted = np.random.randint(len(ls_img_paths))
        #Loading desired images
        img = cv2.imread(ls_img_paths[rd_image_seleted])
        h, w, _ = img.shape

        #Expanding dimension to one sample
        imgs = expand_dims(img, 0)
        datagen = datagenerate()
        batchs = datagen(images=imgs)

        #Remember to convert these images to unsigned integers for viewing 
        image = batchs[0].astype('uint8')
        #Saving the data
        cv2.imwrite(os.path.join(
            output_dir, 
            'aug_{}_{}'.format(
                str(time.time()), 
                os.path.basename(ls_img_paths[rd_image_seleted]))), 
            image
        )


def get_cpu_core(num_workers):
    """Get number of CPU cores
    """
    cpu_core = multiprocessing.cpu_count()//2

    if num_workers == -1: num_workers=cpu_core
    if (num_workers < 0 and num_workers != -1) or num_workers > cpu_core:
        warnings.warn('[WARNING]: num_workers do not match, used num_workers=1 as default')
        num_workers=1

    return num_workers