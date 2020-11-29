#-*-coding:utf-8-*-

# TensorFlow and tf.keras
import tensorflow as tf

#from tensorflow.keras import layers
from tensorflow.keras.layers import *
from tensorflow.keras import Model, Sequential
from tensorflow.keras.models import load_model
from sklearn.utils import Bunch

# Helper libraries
import numpy as np
import matplotlib.pyplot as plt
from classifier.utils import *

class Classifier:
	def __init__(self):
		self.cnn = None
		self.dataset = None
		self.trainset, self.validset, self.testset = None, None, None
		self.history = None

	def build_dataset(self, rawdata_paths, descrpath=None):
		if isinstance(rawdata_paths, str):
			rawdata_paths = [rawdata_paths]

		rawdatas = [np.load(rp) for rp in rawdata_paths]
		self.image_shape = rawdatas[0]["img"][0].shape
		self.sensors_shape = rawdatas[0]["sensors"][0].shape
		self.dataset = Bunch() #sklearn-friendly dataset
		self.dataset.data = np.concatenate([np.asarray(list(zip(rawdata['img'], rawdata['sensors']))) for rawdata in rawdatas])
		self.dataset.targets = np.concatenate([np.asarray(list(rawdata['class_id'])) for rawdata in rawdatas])
		self.dataset.frame = None
		if(descrpath):
			with open(descrpath, 'r') as descr:
				self.dataset.frame = descr.read()
		self.dataset.feature_names = [f"camera_image {self.image_shape}px", [f"laser_sensor_{i} (0-3m)" for i in range(self.sensors_shape[0])]]
		self.dataset.target_names = rawdatas[0]["labels"]

	def build_model(self, summary=True):
		assert self.dataset is not None

		#inputs : images and sensors
		img_input = Input(shape=self.image_shape)
		s_input = Input(shape=self.sensors_shape)
		#images pre-treament
		conv_img_input = Conv2D(32, 5, activation="relu")(img_input)
		max_conv_img = MaxPool2D((4, 4))(conv_img_input)
		flatten = Flatten()(max_conv_img)
		#sensors pre-treament
		sensor_dense = Dense(self.sensors_shape[0]//2)(s_input)
		#merging inputs
		layer = Concatenate()([flatten, sensor_dense])
		#hidden layers
		layer = Dense(512, activation="relu")(layer)
		layer = Dense(32, activation="relu")(layer)
		#ouput layer
		z = Dense(self.dataset.target_names.size, activation="softmax")(layer)

		self.cnn = Model(inputs=[img_input, s_input], outputs=z)

		self.cnn.compile(optimizer='adam',
		              loss='sparse_categorical_crossentropy',
		              metrics=['accuracy'])

		if(summary): self.cnn.summary()

	def fit(self, *args, **kwargs):
		assert self.dataset is not None
		assert self.cnn is not None

		if(self.trainset is None):
			print("no train set has been created, creating train valid and test sets...")
			self.train_valid_test_split()


		options = {"x":[np.asarray(list(zip(*list(self.trainset[0])))[0]), np.asarray(list(zip(*list(self.trainset[0])))[1])],
		"y":self.trainset[1],
		"epochs":6,
		"validation_data":([np.asarray(list(zip(*list(self.validset[0])))[0]), np.asarray(list(zip(*list(self.validset[0])))[1])], self.validset[1]),
		**kwargs
		}

		self.history = self.cnn.fit(**options)

	def evaluate(self, **kwargs):
		options = {"x":[np.asarray(list(zip(*list(self.testset[0])))[0]), np.asarray(list(zip(*list(self.testset[0])))[1])],
		"y":self.testset[1],
		**kwargs
		}
		return self.cnn.evaluate(**options)

	def predict(self, **kwargs):
		options = {"x":[np.asarray(list(zip(*list(self.testset[0])))[0]), np.asarray(list(zip(*list(self.testset[0])))[1])],
		"y":self.testset[1],
		**kwargs
		}
		return self.cnn.predict(**options)

	def train_valid_test_split(self, ratio=(0.4, 0.3, 0.3), **kwargs):
		assert self.dataset is not None

		self.trainset, self.validset, self.testset = data_split(self.dataset.data, self.dataset.targets, ratio, **kwargs)

	def save(self, *args, **kwargs):
		self.cnn.save(*args, **kwargs)

	def load(self, *args, **kwargs):
		self.cnn = load_model(*args, **kwargs)
