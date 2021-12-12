from imgaug import augmenters as iaa
import matplotlib.pyplot as plt
from itertools import cycle
from scipy import interp
import tensorflow as tf

import itertools
import numpy as np
import json
import argparse
import warnings
import os

from synth.utils import datagenerate
from sklearn.metrics import roc_curve, auc, accuracy_score
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix


def rand_augment(images):
    """Random augmentation

    Args:
        images (4D array): data images
         
    Returns:
        [4D array] data after augmentation
    """
    rand_aug = datagenerate()
    images = tf.cast(images, tf.uint8)
    return rand_aug(images=images.numpy())


def get_data(input_dir, fname):
    """Get data from npy file

    Args:
        input_dir (string): ịnput directory
        fname (string): name of npy file
         
    Returns:
        [4D array] data
    """
    f = os.path.join(input_dir, fname + ".npy")
    return np.load(f)


def ensure_dir(directory):
    """Make sure the directory exists

    Args:
        directory (string): name of directory
         
    Returns:
        None
    """
    if not os.path.exists(directory):
        warnings.warn('''[WARNING]: Output directory not found.
            The default output directory will be created.''')
        os.makedirs(directory)


def get_target_names(json_label_decode):
    """Get encode of label

    Args: 
        json_label_decode (string): path to json file

    Returns:
        [dict] encode of label
    """
    with open(json_label_decode) as f:
        label_decode = json.load(f)
    return label_decode


def history_plot(FLAGS, n_class):
    """Plot traning history about: categorical_accuracy, loss and auc, accuracy

    Args:
        FLAGS (argument parser): input information
        n_class (int): number of classes

    Returns:
        [None]
    """
    with open(FLAGS.json_history) as f:
        h = json.load(f)
        
    history = dict()
    if n_class == 2:
        history['accuracy'] = list(h['binary_accuracy'].values())
        history['val_accuracy'] = list(h['val_binary_accuracy'].values())
    else:
        history['accuracy'] = list(h['categorical_accuracy'].values())
        history['val_accuracy'] = list(h['val_categorical_accuracy'].values())
    history['loss'] = list(h['loss'].values())
    history['val_loss'] = list(h['val_loss'].values())
    
    history['auc'] = list(h['auc'].values())
    history['val_auc'] = list(h['val_auc'].values())

    x = np.arange(len(history['accuracy']))
    fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(8, 4))
    fig.suptitle('History training of {}'.format(FLAGS.model_name))

    #plot accuracy
    ax[0].plot(x, history['accuracy'], label = 'accuracy')
    ax[0].plot(x, history['val_accuracy'], label = 'val_accuracy')
    ax[0].plot(x, history['auc'], label = 'auc')
    ax[0].plot(x, history['val_auc'], label = 'val_auc')
    ax[0].set_title("accuracy/auc")
    ax[0].set_xlabel('epoch')
    ax[0].set_ylabel('accuracy/auc')
    ax[0].legend(shadow=True, fancybox=True, loc='lower right')

    #plot loss
    ax[1].plot(x, history['loss'], label = 'loss')
    ax[1].plot(x, history['val_loss'], label = 'val_loss')
    ax[1].set_title("loss")
    ax[1].set_xlabel('epoch')
    ax[1].set_ylabel('loss')
    ax[1].legend(shadow=True, fancybox=True, loc='upper right')

    plt.savefig(os.path.join(FLAGS.output_dir, 'history_of_{}.png'.format(FLAGS.model_name)))
    plt.close()


def roc_plot(FLAGS, y_test, y_score, target_names):
    """Plot Receiver Operating Characteristic curve

    Args:
        FLAGS (argument parser): input information
        y_test (2D array): true label of test data
        y_score (2D) array: prediction label of test data
        target_names (1D array): array of encode label

    Returns:
        [datagen]
    """
    n_classes = y_test.shape[1]
    lw = 2
    # Compute ROC curve and ROC area for each class
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    for i in range(n_classes):
        fpr[i], tpr[i], _ = roc_curve(y_test[:, i], y_score[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])

    # First aggregate all false positive rates
    all_fpr = np.unique(np.concatenate([fpr[i] for i in range(n_classes)]))

    # Then interpolate all ROC curves at this points
    mean_tpr = np.zeros_like(all_fpr)
    for i in range(n_classes):
        mean_tpr += interp(all_fpr, fpr[i], tpr[i])

    # Finally average it and compute AUC
    mean_tpr /= n_classes

    # Plot all ROC curves
    plt.figure()

    # colors = cycle(['aqua', 'darkorange', 'cornflowerblue'])
    for i in range(n_classes):
        plt.plot(fpr[i], tpr[i], lw=lw,
                 label='ROC curve of class {0} (area = {1:0.2f})'
                 ''.format(target_names[i], roc_auc[i]))

    plt.plot([0, 1], [0, 1], 'k--', lw=lw)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic - {}'.format(FLAGS.model_name))
    plt.legend(loc="lower right")
    plt.savefig(os.path.join(FLAGS.output_dir, 'roc_of_{}.png'.format(FLAGS.model_name)))
    plt.close()


def statistics(FLAGS, y_test, target_names):
    """Statistics for the number of images per class after shuffling.

    Args:
        FLAGS (argument parser): input information
        y_test (2D array): true label of test data
        target_names (1D array): array of encode label

    Returns:
        [None]
    """
    sta_test = np.sum(y_test, axis=0); temp = sta_test.copy()
    #sum equal one
    sta_test = sta_test/y_test.shape[0]
    explode = np.ones(len(sta_test))*0.1
    #label
    target_names = [(name + ':' + str(int(item))) for name, item in zip(target_names, temp)]
    #plot
    plt.pie(sta_test, explode=explode, labels=target_names, shadow=True, startangle=45)
    plt.axis('equal')
    plt.legend(title='Statistic On Test Data')
    plt.savefig(os.path.join(FLAGS.output_dir, 'label_statistics.png'))
    plt.close()


def plot_confusion_matrix(FLAGS, cm, classes,
                          normalize=True,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """This function prints and plots the confusion matrix.
       Normalization can be applied by setting `normalize=True`.

    Args:
        cm (2D array): confusion matrix
        classes (1D list): list of class names
        normalize (boolean): normalization or not, default true
    
    Returns:
        [None]
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1, keepdims = True)

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')

    plt.savefig(os.path.join(FLAGS.output_dir, 'confusion_matrix_of_{}.png'.format(FLAGS.model_name)))
    plt.close()


def evaluate(FLAGS, y_test, y_score, n_class):
    """Evaluate the quality of the model

    Args:
        FLAGS (argument parser): input information
        y_test (2D array): true label of test data
        y_score (2D) array: prediction label of test data
        n_class (int): number of classes

    Returns:
        [None]
    """
    if os.path.exists(FLAGS.json_label_decode):
        label_decode = get_target_names(FLAGS.json_label_decode)

        target_names = []
        for i in label_decode:
            target_names.append(label_decode[i])
    else: raise ValueError('[ERROR]: {} is not found'.format(FLAGS.json_label_decode))

    #statistics
    print("[INFOR]: Plot Statistics\n")
    statistics(FLAGS, y_test, target_names)

    #plot roc
    print("[INFOR]: Plot Receiver Operating Characteristic\n")
    roc_plot(FLAGS, y_test, y_score, target_names)

    #plot history
    print("[INFOR]: Plot History.\n")
    if os.path.exists(FLAGS.json_history):
        history_plot(FLAGS, n_class)
    else: warnings.warn('''[WARNING]: {} is not found, 
        plot history is ignored'''.format(FLAGS.json_history))

    #convert to 1D array
    y_test = np.argmax(y_test, axis=1)
    y_score = np.argmax(y_score, axis=1)

    print("\n\n[INFOR]: Plot confusion matrix\n")
    cm = confusion_matrix(y_test, y_score)
    plot_confusion_matrix(FLAGS, cm, target_names, normalize=False)

    print("\n\n[INFOR]: Report test data\n")
    print(classification_report(y_test, y_score, 
        target_names=target_names))

    print("\n\n[INFOR]: Report accuracy\n")
    print(accuracy_score(y_test, y_score), '\n')

    print("\n\n[INFOR]: See more result in {} folder\n".format(FLAGS.output_dir))

    print()
