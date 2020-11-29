import os

PATH = os.path.dirname(__file__)
DATASET_FOLDER_PATH = os.path.join(PATH, 'datasets')

SAMPLE_DATASET_PATH = os.path.join(DATASET_FOLDER_PATH, 'sample_dataset.npz')
GENERATED_DATASET_PATH = os.path.join(DATASET_FOLDER_PATH, 'dataset.npz')

DATASET_PATHS = [os.path.join(DATASET_FOLDER_PATH, f'dataset_{i}-{i+1999}.npz')
                  for i in range(0, 10000, 2000)]
CNNS = os.path.join(PATH, 'cnns')
