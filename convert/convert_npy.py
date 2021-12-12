from utils import convert_to_npy_custom
from utils import convert_to_npy
import tensorflow as tf
import numpy as np
import argparse
import os

os.environ['TF_XLA_FLAGS'] = '--tf_xla_enable_xla_devices'

def main(FLAGS):
	if FLAGS.custom_split_data: convert_to_npy_custom(FLAGS.input_dir, FLAGS.output_dir, FLAGS.input_shape)
	else: convert_to_npy(FLAGS.input_dir, FLAGS.output_dir, FLAGS.input_shape)
	return 0


if __name__ == '__main__':

	FLAGS = None
	parser = argparse.ArgumentParser()
	parser.add_argument(
		'--input_dir',
		type=str,
		default='../database/dataset',
		help='Input data directory'
	)
	parser.add_argument(
		'--output_dir',
		type=str,
		default='dataset',
		help='Output data directory'
	)
	parser.add_argument(
		'--custom_split_data',
		type=bool,
		default=True,
		help='''
		The required directory structure must be organized
		as readme file.
        NOTE THAT: label of images == folder name containing images

		'''
	)
	parser.add_argument(
		'--input_shape',
		type=tuple,
		default=(200, 260), #HxW
		help='Shape of input, HxW'
	)
	parser.add_argument(
		'--n_class',
		type=int,
		default=2,
		help='number of classes'
	)
	FLAGS = parser.parse_args()
	print("input_dir = ", FLAGS.input_dir)
	print("output_dir = ", FLAGS.output_dir)
	print("custom_split_data = ", FLAGS.custom_split_data)
	print("input_shape = ", FLAGS.input_shape)
	print("n_class = ", FLAGS.n_class)

	main(FLAGS)