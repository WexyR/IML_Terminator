import random
import time
import numpy as np
import resources
import os
import cv2

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
        shape = (len(img), len(img[0]))
        ratio = 25 / 100
        img = cv2.resize(img, (int(shape[1] * ratio), int(shape[0] * ratio)), interpolation=cv2.INTER_AREA)

        sensors = self.__agent.getFrontLaserValue()
        cube.remove()
        return (img, sensors, cubeRGB, cubeSize)

    def genDataset(self, output_path, size, seed=None):
        r = random.Random(seed)
        img=[]
        sensors=[]
        cubeRGB=[]
        cubeSize=[]
        for i in range(size):
            print(f"\rGenerating data... [{'='*int(20*i/size)}>{' '*int(20-20*i/size)}] {i}/{size}",end='')

            line = self.genClassData(r.random() + 0.5,  # CubeRho
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
