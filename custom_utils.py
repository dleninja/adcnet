"""
Automated Dispersion Compensation Network (ADC-Net).

This file is stores several utility functions, namely:
- load_multichannel_image
- SSIMloss

@author: dleninja
"""
#
import tensorflow as tf
#
import numpy as np
import pandas as pd
import os
from skimage import img_as_float, transform, exposure, io, color
from pathlib import Path
from matplotlib import image
import matplotlib.pyplot as plt
import cv2
#
def load_multichannel_image(df, im_shape, path_list, column):
    """
    Custom multi-channal image reader.

    Args:
        df: dataframe, contains the image name with extension, e.g. "image001.png"
        im_shape: tuple, containing the height and width of the images.
        path_list: list of Paths, for all the different directories containing the different inputs.
        column: integer, the specific column to read the dataframe from.

    Returns:
        Outputs a multi-dimension array, of size (len(df), im_shape[0], im_shape[1], len(path_list)).
    """
    #
    images = []
    n_channels = len(path_list)
    #
    for i, item in df.iterrows():
        #
        image = np.zeros([im_shape[0], im_shape[1], n_channels])
        #
        for j in range(n_channels):
            temp_item = path_list[j] / item[column]
            temp_image = io.imread(temp_item)
            temp_image = temp_image / 255
            #
            if j == 0:
                temp_im_shape = temp_image.shape
            #
            image[0:temp_im_shape[0],0:temp_im_shape[1],j] = temp_image
            #
        #
        images.append(image)
        #
    images = np.array(images)
    print(' --- Images loaded --- ');
    print('\t{}'.format(images.shape))
    #
    return images
#
def SSIMLoss(y_true, y_pred):
    """
    Structural similiary index measure (SSIM) loss function

    Args:
        y_true: ground truth tensor.
        y_pred: prediction tensor by the model.

    Returns:
        loss.
    """
    return (1 - tf.reduce_mean(tf.image.ssim(y_true, y_pred, 1.0)))
