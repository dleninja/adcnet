"""
Automated Dispersion Compensation Network (ADC-Net).

This file contains the network architecture of ADC-Net. All functions and dependencies are self-contained

@author: dleninja
"""
#
from tensorflow.keras.layers import Input, Conv2D, ZeroPadding2D, BatchNormalization, UpSampling2D, Concatenate, Activation, AveragePooling2D, MaxPool2D
from tensorflow.keras import Model
from tensorflow.keras import backend
import sys

def adcnet_model(block=[6, 12, 24, 16], height=None, width=None, n_channels=3):
	"""
	ADC-Net model is an encoder-decoder network.
	- The encoder is based on the DenseNet-121 architecture.
	- The decoder is a custom a decoder.

	Args:
		blocks: number of blocks for the four dense layers
		input_size: shape tuple, if specified, should not be smaller than 32.
		n_channels: number of magnitude channels

	Returns:
		Model of network.
	"""
	input_size = (height, width, n_channels)
	#
	bn_axis = 3 if backend.image_data_format() == 'channels_last' else 1
	inputs = Input(input_size, name="Input_Layer")
	x = ZeroPadding2D(padding=(3,3), name="conv1_zeropad1")(inputs)
	x = Conv2D(filters=64, kernel_size=7, strides=2, use_bias=False, padding="valid", name="conv1_conv")(x)
	x = BatchNormalization(axis=bn_axis, epsilon=1.001e-5, name="conv1_bn")(x)
	x0 = Activation("relu", name="conv1_relu")(x)
	x = ZeroPadding2D(padding=(1,1), name="conv1_zeropad2")(x0)
	x = MaxPool2D(pool_size=3, strides=2, name="pool1")(x)
	#
	x = dense_block(x, block[0], name="conv2")
	[x, x1] = transition_block(x, 0.5, name="pool2")
	x = dense_block(x, block[1], name="conv3")
	[x, x2] = transition_block(x, 0.5, name="pool3")
	x = dense_block(x, block[2], name="conv4")
	[x, x3] = transition_block(x, 0.5, name="pool4")
	x = dense_block(x, block[3], name="conv5")
	#
	x = BatchNormalization(axis=bn_axis, epsilon=1.001e-5, name="bn")(x)
	x = Activation("relu", name='relu')(x)
	#
	x = UpSampling2D(size=(2,2))(x)
	x = decoder_block(x, x3, 256, name="decode1")
	x = UpSampling2D(size=(2,2))(x)
	x = decoder_block(x, x2, 128, name="decode2")
	x = UpSampling2D(size=(2,2))(x)
	x = decoder_block(x, x1, 64, name="decode3")
	x = UpSampling2D(size=(2,2))(x)
	x = decoder_block(x, x0, 32, name="decode4")
	#
	x = UpSampling2D(size=(2,2))(x)
	x = Conv2D(filters=16, kernel_size=3, strides=1,  use_bias=False, padding="same", name="decoder_conv1_conv")(x)
	x = BatchNormalization(axis=bn_axis, epsilon=1.001e-5, name="decoder_conv1_bn")(x)
	x = Activation("relu", name="decoder_conv1_relu")(x)
	x = Conv2D(filters=16, kernel_size=3, strides=1,  use_bias=False, padding="same", name="decoder_conv2_conv")(x)
	x = BatchNormalization(axis=bn_axis, epsilon=1.001e-5, name="decoder_conv2_bn")(x)
	x = Activation("relu", name="decoder_conv2_relu")(x)
	#
	x = Conv2D(filters=1, kernel_size=1, strides=1, use_bias=True, padding="same", name="final_conv")(x)
	outputs = Activation("sigmoid", name="output_classification")(x)
	#
	adcnet = Model(inputs, outputs)
	#
	return adcnet

def dense_block(x, blocks, name):
	"""
	Densely connected blocks

	Args:
		x: input tensor.
		blocks: integer, the number of conv blocks.
		name: string, block label.

	Returns:
		x: Output tensor for the block.
	"""
	for i in range(blocks):
		x = conv_block(x, 32, name=name + "_block" + str(i + 1))
	
	return x

def transition_block(x, reduction, name):
	"""
	Transition Block

	Args:
		x: input tensor.
		reduction: float, compression rate at transition layers.
		name: string, block label.

	Returns:
		x1: Output tensor for the block.
		x2: Conv2D output tensor for skip-connection with the decoder
	"""
	bn_axis = 3 if backend.image_data_format() == "channels_last" else 1
	x1 = BatchNormalization(axis=bn_axis, epsilon=1.001e-5, name=name + "_bn")(x)
	x1 = Activation('relu', name=name + "_relu")(x1)
	x2 = Conv2D(filters=int(backend.int_shape(x)[bn_axis]*reduction), kernel_size=1, use_bias=False, name=name + "_conv")(x1)
	x1 = AveragePooling2D(2, strides=2, name=name + "_pool")(x2)

	return x1, x2

def decoder_block(x, x1, growth_rate, name):
	"""
	Decoder Block

	Args:
		x: input tensor 1, tensor from previous upsampled layer.
		x1: input tensor 2, tensor from corresponding layer in the encoder.
		growth_rate: float, growth rate for decoder block.
		name: string, block name.

	Returns:
		x2: Output tensor for the block.
	"""
	bn_axis = 3 if backend.image_data_format() == 'channels_last' else 1
	x2 = Concatenate(axis=bn_axis, name=name + "_concat")([x, x1])
	#
	x2 = Conv2D(filters=growth_rate, kernel_size=3, use_bias=False, padding="same", name=name + '_0_conv')(x2)
	x2 = BatchNormalization(axis=bn_axis, epsilon=1.001e-5, name=name + "_0_bn")(x2)
	x2 = Activation("relu", name=name + '_0_relu')(x2)
	#
	x2 = Conv2D(filters=growth_rate, kernel_size=3,  use_bias=False, padding="same", name=name + '_1_conv')(x2)
	x2 = BatchNormalization(axis=bn_axis, epsilon=1.001e-5, name=name + "_1_bn")(x2)
	x2 = Activation("relu", name=name + '_1_relu')(x2)

	return x2

def conv_block(x, growth_rate, name):
	"""
	Building block for the Dense block.

	Args:
		x: input tensor.
		growth_rate: float, growth rate at dense layers.
		name: string, block label.

	Returns:
		x2: Output tensor for the block.
	"""
	bn_axis = 3 if backend.image_data_format() == 'channels_last' else 1
	x1 = BatchNormalization(axis=bn_axis, epsilon=1.001e-5, name=name + "_0_bn")(x)
	x1 = Activation("relu", name=name + '_0_relu')(x1)
	#
	x1 = Conv2D(filters=4*growth_rate, kernel_size=1, use_bias=False, name=name + '_1_conv')(x1)
	x1 = BatchNormalization(axis=bn_axis, epsilon=1.001e-5, name=name + "_1_bn")(x1)
	x1 = Activation("relu", name=name + '_1_relu')(x1)
	#
	x1 = Conv2D(filters=growth_rate, kernel_size=3, use_bias=False, padding="same", name=name + "_2_conv")(x1)
	x2 = Concatenate(axis=bn_axis, name=name + "_concat")([x, x1])

	return x2
