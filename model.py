from tensorflow.keras.layers import Input, Conv2D, ZeroPadding2D, BatchNormalization, UpSampling2D, Concatenate, Activation
from tensorflow.keras import Model

def model(input_size = (None, None, n_channels)):
  inputs = Input(input_size, name="InputLayer")
  final_conv = Conv2D(filters=3, kernel_size=1, strides=1, activation="sigmoid", padding="same", name="OutputLayer")(inputs)
  
  model = Model(inputs, final_conv)
