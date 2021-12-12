from imutils.paths import list_images
from utils import submit_thread
from utils import get_cpu_core
from utils import ensure_dir

import numpy as np
import argparse
import tensorflow as tf
import warnings
import os


def main(FLAGS):
    #get argument
    num_workers = get_cpu_core(FLAGS.num_workers)

    #number of images per core
    num_imgs = FLAGS.num_imgs//num_workers

    #check
    if os.path.exists(FLAGS.input_dir): 
        ensure_dir(FLAGS.output_dir)
        ls_img_paths = list(list_images(FLAGS.input_dir))
        submit_thread(ls_img_paths, FLAGS.output_dir, num_imgs, num_workers)
    else: 
        raise ValueError('[Error]: {} not found'.format(FLAGS.input_dir))


if __name__ == '__main__':

    FLAGS = None
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--input_dir',
        type=str,
        default='./input-synth/',
        help='Input data directory'
    )
    parser.add_argument(
        '--output_dir',
        type=str,
        default='./output-synth/',
        help='Output data directory'
    )
    parser.add_argument(
        '--num_imgs',
        type=int,
        default=20,
        help='Number of image to synth data.'
    )
    parser.add_argument(
        '--num_workers',
        type=int,
        default=-1,
        help=
        '''Maximum number of processes to spin up 
           when using process-based threading.
           Default = -1, use all core of CPU
        '''
    )
    FLAGS = parser.parse_args()
    print("\ninput_dir = ", FLAGS.input_dir)
    print("output_dir = ", FLAGS.output_dir)
    print("num_imgs = ", FLAGS.num_imgs)
    print("num_workers = ", FLAGS.num_workers, '\n')
    main(FLAGS)