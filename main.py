"""
Automated Dispersion Compensation Network (ADC-Net).

This file contains the code to train ADC-Net. For this example, a 5 input model will be demonstrated.
Minor modifications is needed for the other types of input models.

@author: dleninja
"""
#
import tensorflow as tf
#
"""
For machines with dedicated GPU(s), utilize the GPU for tensorflow training
"""
gpus = tf.config.experimental.list_physical_devices("GPU")
print("Num GPUs Available: ", len(gpus))
#
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)
#
from tensorflow.keras.applications import DenseNet121
from tensorflow.keras.layers import Input
from tensorflow.keras import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras import metrics
from tensorflow.keras.callbacks import ModelCheckpoint
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
from model import *
from custom_utils import *
#
"""
Load the ADCNet model, dependent on the functions defined in model.py
"""
#
model = adcnet_model(block=[6, 12, 24, 16], height=608, width=320, n_channels=5)
model.summary()
#
"""
Import the DenseNet121 pre-trained weights from the ImageNet dataset into the encoder of ADCNet
"""
#
densenet_model = tf.keras.applications.DenseNet121(
    include_top=True,
    weights="imagenet",
    input_tensor=None,
    input_shape=None,
    pooling=None,
    classes=1000,
    )
weights = [layer.get_weights() for layer in densenet_model.layers[5:427]]
for layer, weight in zip(model.layers[5:427], weights):
    layer.set_weights(weight)
#
"""
Compile the Model, dependent on the loss function defined in custom_utils.py
"""
#
model.compile(
    optimizer = Adam(learning_rate=0.0001),
    loss = SSIMLoss,
    metrics = ["acc"]
)
#
batch_size = 16
#
export_dir = Path("Results")
if not os.path.exists(export_dir):
    os.makedirs(export_dir)
#
model_file_format = os.path.join(
    export_dir, 
    "dispersion_model.{epoch:03d}.hdf5"
)
checkpointer = ModelCheckpoint(
    model_file_format,
    period = 1,
    save_best_only=True,
    save_weights_only=True
)
#
"""
Import the dataset. For our implementation purposes, we will be directly loading the data by a custom imageloader
Dependent on the function in custom_utils.py
"""
#
df = pd.read_csv("train.csv")
n_train = int(len(df)*0.8)
#
df_train = df[:n_train]
df_train = df_train.sample(frac=1)
#
df_valid = df[n_train:]
df_valid = df_valid.sample(frac=1)
#
path1 = Path("dataset/magnitude1")
path2 = Path("dataset/magnitude3")
path3 = Path("dataset/magnitude5")
path4 = Path("dataset/magnitude7")
path5 = Path("dataset/magnitude9")
path6 = Path("dataset/compensated")
#
path_list_X = [path1, path2, path3, path4, path5]
path_list_y = [path6]
#
im_shape = (608, 320)
#
X_train = load_multichannel_image(df_train, im_shape, path_list_X, 0)
y_train = load_multichannel_image(df_train, im_shape, path_list_y, 0)
#
X_valid = load_multichannel_image(df_valid, im_shape, path_list_X, 0)
y_valid = load_multichannel_image(df_valid, im_shape, path_list_y, 0)
#
"""
Train the model
"""
#
model.fit(X_train, y_train, batch_size, 
    steps_per_epoch = n_train // batch_size,
    validation_data = (X_valid, y_valid),
    callbacks = [checkpointer],
    epochs = 10)
