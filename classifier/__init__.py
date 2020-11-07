#-*-coding:utf-8-*-

# TensorFlow and tf.keras
import tensorflow as tf

#from tensorflow.keras import layers
from tensorflow.keras.layers import *
from tensorflow.keras import Model, Sequential

# Helper libraries
import numpy as np
import matplotlib.pyplot as plt

class Classifier:
	def __init__(self, dataset_path, **cnn_kwargs):
		dataset = np.load(dataset_path)
		self.data = {"X":dataset['img'],
					"y":(dataset['cubeSize']-0.5)/1.5}
		print(self.data["X"][0], self.data["y"][0])
		# self.cnn = self.__build_model(self.data["X"][0][0].shape, self.data["X"][0][1].shape, hidden_layers)
		print(self.data["X"][0].shape)
		self.__build_model(self.data["X"][0].shape)
		self.cnn.summary()
		self.cnn.fit(self.data["X"][:70], self.data["y"][:70], epochs=5)
		self.cnn.evaluate(self.data["X"][70:], self.data["y"][70:], verbose=2)



	def __build_model(self, image_shape):

		# s_input = Input(shape=sensors_shape)
		# conv_img_input = Conv2D(32, 5, activation="relu", input_shape=image_shape)
		# max_conv_img = MaxPool2D((2, 2))(conv_img_input)
		# conv_max = Conv2D(32, 5, activation="relu")(max_conv_img)
		# max_conv_max = MaxPool2D((2, 2))(conv_max)
		# flatten = Flatten()(max_conv_max)
		# layer = Concatenate()([flatten, s_input])
		# for hl in hidden_layers:
		# 	layer = Dense(hl, activation="relu")(layer)
		# 	# layer = Dropout(0.2)(layer)
		# output = Dense(4, activation="sigmoid")
		print("set up CNN")
		self.cnn = Sequential([
	  			Conv2D(32, 9, activation='relu',input_shape=image_shape),
	  			MaxPool2D((2,2)),
	  			# Conv2D(32, 9, activation='relu'),
	  			# MaxPool2D((2,2)),
	  			Flatten(),
	  			# Dense(128, activation='relu'),
	  			# Dropout(0.2),
	  			Dense(12, activation='relu'),
	  			Dense(1, activation='sigmoid')
		])

		self.cnn.compile(optimizer='adam',
		              loss='mean_absolute_error')
