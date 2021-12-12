from imutils.paths import list_images
from tqdm import tqdm

from sklearn.model_selection import GridSearchCV
from sklearn.neighbors import KernelDensity

import matplotlib.pyplot as plt
import numpy as np

import warnings
import imagesize
import random
import os
import cv2
import argparse

'''
The program helps us analyze the input shape so that it can give the appropriate input shape.
'''

def kde_plot(P, fname, num_point, output_dir):
    """Plot 2D Probability Density Function

    Args:
        P (pdf): probability density function
        fname (string): weight | high
        num_point (int): number of data points
        output_dir (string): output folder

    Returns:
        [None]
    """
    #define domain value to plot
    x_plot = np.arange(-20, (num_point+20), 1)
    x_plot = x_plot.reshape((x_plot.shape[0], 1))
    # print(x_plot.shape)

    #calculate the density value
    log_dens = np.array(P.score_samples(x_plot))

    fig, ax = plt.subplots()
    y_plot = np.exp(log_dens)
    index = np.argmax(y_plot)
  
    ax.plot(y_plot, lw=2,
              linestyle='-', label='Highest density of {} is: {}'.format(fname, index))

    X_plot = np.arange(-20, (num_point+20), 1)
    ax.fill(X_plot, np.exp(log_dens), fc='blue', alpha=0.2)

    ax.legend(loc='upper left')
    plt.savefig(os.path.join(output_dir, 'pdf_{}.png'.format(fname)))

    return 0


def get_kde(x):
  """Use Kernel Density Estimation to estimate Probability Density Function of data x

    Args:
        x (2D array): datas
  
    Returns:
        [pdf object] Probability Density Function
    """
  # use grid search cross-validation to optimize the bandwidth
  params = {'bandwidth': np.logspace(1, 10, 10)}
  grid = GridSearchCV(KernelDensity(), params)
  grid.fit(x)
  # print("best bandwidth: {0}".format(grid.best_estimator_.bandwidth))

  # use the best estimator to compute the kernel density estimate
  kde = grid.best_estimator_
  return kde


def analysis(input_dir, output_dir):
  """Analysis of input shape

    Args:
        input_dir (string): path to data folder
  
    Returns:
        [None]
    """
  img_paths = list(list_images(input_dir))
  N = len(img_paths)
  MAX=10000
  print('[INFOR]: Has found {} images'.format(N))

  if N>MAX:
    print('[WARNING]: The number of images is relatively large for analysis -> time consuming')
    print('[INFOR]: The program will randomly select 20% of the images for analysis.')

    img_paths = random.choices(img_paths, k=int(0.2*N))

  #get weight, high of list image
  print('[INFOR]: Get image size information of {} images'.format(len(img_paths)))
  imgs_size = []
  for i in tqdm(range(len(img_paths))):
    width, height = imagesize.get(img_paths[i])
    imgs_size.append([width, height])

  imgs_size = np.array(imgs_size)

  #separation
  x = imgs_size[:, 0]
  y = imgs_size[:, 1]

  #standardized
  x = (np.asarray(x, dtype=int)).reshape((x.shape[0], 1))
  y = (np.asarray(y, dtype=int)).reshape((y.shape[0], 1))

  #get dke
  print('[INFOR]: Analyzing Probability Density Function')
  print('[INFOR]: ...')
  P1 = get_kde(x)
  P2 = get_kde(y)

  #plot
  x_num_point = x.max()
  y_num_point = y.max()
  kde_plot(P1, 'width', x_num_point, output_dir)
  kde_plot(P2, 'height', y_num_point, output_dir)
  print('[Infor]: Complete analysis, see result at: {}'.format(output_dir))

  return 0


def main(FLAGS):
	analysis(FLAGS.input_dir, FLAGS.output_dir)


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
		default='../database/output',
		help='Output data directory'
	)
	FLAGS = parser.parse_args()
	print("input_dir = ", FLAGS.input_dir)
	print("output_dir = ", FLAGS.output_dir)

	main(FLAGS)