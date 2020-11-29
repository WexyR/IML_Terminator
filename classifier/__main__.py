from classifier import *
import resources
import os, sys

if __name__ == '__main__':
	if(len(sys.argv) != 2):
		raise ValueError("Please specify if you want to fit a new model (1:yes, 0:no)")
	print(sys.argv)
	if(int(sys.argv[1])):
		cnn = Classifier()
		cnn.build_dataset(resources.DATASET_PATHS)
		cnn.build_model()
		cnn.fit()
		cnn.evaluate()
		cnn.save(resources.CNNS)
	else:
		cnn = Classifier()
		cnn.load(resources.CNNS)
		cnn.build_dataset(resources.DATASET_PATHS)
		cnn.train_valid_test_split()
		cnn.evaluate()
