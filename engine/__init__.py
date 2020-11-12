import random
import time
import numpy as np
import matplotlib.pyplot as plt
import os
import cv2
import pybullet as p

from qibullet import SimulationManager, Camera
from qibullet import PepperVirtual
from engine.cube import Cube
from engine.colors import Colors

class Engine:
    def __init__(self):
        self.__simulation_manager = SimulationManager()
        self.__client_id = self.__simulation_manager.launchSimulation(gui=False)
        self.__agent = self.__simulation_manager.spawnPepper(
            self.__client_id,
            spawn_ground_plane=True)
        self.__laser_handle = self.__agent.subscribeLaser()
        self.__camera_handle = self.__agent.subscribeCamera(PepperVirtual.ID_CAMERA_BOTTOM)

    def genClassData(self, cube_rho, cube_teta, cube_rot, cubeRGB, cubeSize):
        cube = Cube((np.cos(cube_teta)*cube_rho, np.sin(cube_teta)*cube_rho, 0.125*cubeSize),
                    (cube_rot, 0, 0),
                    cubeRGB,
                    cubeSize)

        # Skipping a frame before saving
        img = Camera._getCameraFromHandle(self.__camera_handle).frame
        for i in range(2):
            tmp = img
            while img is tmp:
                time.sleep(0.01)
                img = Camera._getCameraFromHandle(self.__camera_handle).frame

        img = img.copy()
        shape = (len(img), len(img[0]))
        ratio = 25 / 100
        img = cv2.resize(img, (int(shape[1] * ratio), int(shape[0] * ratio)), interpolation=cv2.INTER_AREA)

        sensors = self.__agent.getFrontLaserValue()
        cube.remove()
        return (np.array(img, dtype=np.uint8),
                         np.array(sensors, dtype=np.float16),
                         np.array(cubeRGB, dtype=np.uint8),
                         np.array(cubeSize, dtype=np.float16),
                         np.array(Colors.classifiate(cubeRGB), dtype=np.uint8))

    def genDataset(self, output_path, size, seed=None, debug=False):
        r = random.Random(seed)
        img=[]
        sensors=[]
        cubeRGB=[]
        cubeSize=[]
        class_id=[]

        i = 1
        time_start = time.time()
        last_step = 0.0
        av_speed = 1.0
        sensors_fail_count = 0
        while i<=size:
            av_speed = 0.90*av_speed+0.1/(time.time()-last_step)
            last_step = time.time()
            eta = (time.time()-time_start)/i*(size-i)
            print(f"\rGenerating data... [{'='*int(20*i/size)}>{' '*int(20-20*i/size)}] {i}/{size}, "
                  f"ETA {int(eta/60)}min{int(eta%60)}, "
                  f"Current speed is {round(av_speed,2)} per second, "
                  f"Sensors fail rate is {round(sensors_fail_count/i*100,2)}%",end='')

            line = self.genClassData(r.random() + 0.5,  # CubeRho
                                      r.random() * np.pi / 3 - np.pi / 6,  # CubeTeta
                                      r.random() * 2 * np.pi,  # CubeRotZ
                                      (r.random(), r.random(), r.random()),  # RGB
                                      r.random() * 1.5 + 0.5)
            sensors_ok = False
            for j in line[1]:
                if j != 3.0:
                    sensors_ok = True
                    break

            if not sensors_ok:
                sensors_fail_count+=1
                continue

            if debug:
                plt.imshow(line[0], interpolation="bilinear")
                plt.show()

            i+=1
            img+=[line[0]]
            sensors+=[line[1]]
            cubeRGB+=[line[2]]
            cubeSize+=[line[3]]
            class_id+=[line[4]]
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        np.savez(output_path,
                 img=img,
                 sensors=sensors,
                 cubeRGB=cubeRGB,
                 cubeSize=cubeSize,
                 class_id=class_id,
                 labels=[color.name for color in Colors])

    def stop(self):
        self.__simulation_manager.stopSimulation(self.__client_id)
