import numpy as np
import resources
from engine import Engine
import os

if __name__ == "__main__":
    dataset_path = os.path.join(resources.DATASETS, 'dataset.npz')
    if not os.path.exists(dataset_path):
        print('starting generating dataset...')
        engine = Engine()
        engine.genDataset(dataset_path, 10000)
        engine.stop()
        print(f"generated and saved dataset at '{dataset_path}'")

    dataset=np.load(dataset_path)
    print(f"loaded previously generated dataset at '{dataset_path}'")

    X = list(zip(dataset['img'], dataset['sensors']))
    y = list(zip(dataset['class_id'], dataset['cubeSize']))
    print(f"first element of the database is:\nX: {X[0]}\ny: {y[0]}\n"
          f"Labels are : {dataset['labels']}")
