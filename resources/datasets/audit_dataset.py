from resources import GENERATED_DATASET
import numpy as np
import matplotlib.pyplot as plt

data = np.load(GENERATED_DATASET)
plt.hist(data['cubeSize'], bins=np.arange(0.0,3.0,0.2), color='r')
plt.hist([size for size, sensors in zip(data['cubeSize'], data['sensors']) if np.sum(sensors)!=len(sensors)*3.0],
         bins=np.arange(0.0,3.0,0.2), color='b')
plt.show()
