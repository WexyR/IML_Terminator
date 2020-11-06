import random
import time
import numpy as np
import resources
import os
import sys

from qibullet import SimulationManager
from qibullet import PepperVirtual
from engine.cube import Cube

class Engine:
    def __init__(self):
        self.__simulation_manager = SimulationManager()
        self.__client_id = self.__simulation_manager.launchSimulation(gui=False)
        self.__agent = self.__simulation_manager.spawnPepper(
            self.__client_id,
            spawn_ground_plane=True)

        self._camera_handle = self.__agent.subscribeCamera(PepperVirtual.ID_CAMERA_BOTTOM)

    def genClassData(self, cube_rho, cube_teta, cube_rot, cubeRGB, cubeSize):
        cube = Cube((np.cos(cube_teta)*cube_rho, np.sin(cube_teta)*cube_rho, 0.125*cubeSize),
                    (cube_rot, 0, 0),
                    cubeRGB,
                    cubeSize)
        time.sleep(0.07)
        img = self.__agent.getCameraFrame(self._camera_handle)
        sensors = self.__agent.getFrontLaserValue()
        cube.remove()
        return (img, sensors, cubeRGB, cubeSize)

    def genDataset(self, output_path, size, nthreads=1, seed=None):
        r = random.Random(seed)
        img=[]
        sensors=[]
        cubeRGB=[]
        cubeSize=[]
        for i in range(size):
            print(f"\rGenerating data... [{'='*int(20*i/size)}>{' '*int(20-20*i/size)}] {i}/{size}",end='')

            line = engine.genClassData(r.random() + 0.5,  # CubeRho
                                      r.random() * np.pi / 3 - np.pi / 6,  # CubeTeta
                                      r.random() * 2 * np.pi,  # CubeRotZ
                                      (r.random(), r.random(), r.random()),  # RGB
                                      r.random() * 1.5 + 0.5)
            img+=[line[0]]
            sensors+=[line[1]]
            cubeRGB+=[line[2]]
            cubeSize+=[line[3]]
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        np.savez(output_path, img=img, sensors=sensors, cubeRGB=cubeRGB, cubeSize=cubeSize)

    def stop(self):
        self.__simulation_manager.stopSimulation(self.__client_id)

if __name__ == "__main__":

    dataset_path = os.path.join(resources.DATASETS, 'dataset.npz')
    if not os.path.exists(dataset_path):
        print('starting generating dataset...')
        engine = Engine()
        engine.genDataset(dataset_path, 1000)
        engine.stop()
        print(f"generated and saved dataset at '{dataset_path}'")

    dataset=np.load(dataset_path)
    print(f"loaded previously generated dataset at '{dataset_path}'")

    X = list(zip(dataset['img'], dataset['sensors']))
    y = list(zip(dataset['cubeRGB'], dataset['cubeSize']))
    print(f"first element of the database is:\nX: {X[0]}\ny: {y[0]}")


