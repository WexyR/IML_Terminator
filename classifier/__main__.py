from classifier import *
import resources
import os

if __name__ == '__main__':
	dataset_path = os.path.join(resources.DATASETS, 'dataset.npz')
	ai = Classifier(dataset_path)
