import random
import time
import numpy as np

from qibullet import SimulationManager
from qibullet import PepperVirtual
from engine.cube import Cube
from matplotlib import pyplot as plt

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
        return {'X':(img, sensors),'Y':(cubeRGB, cubeSize)}

if __name__ == "__main__":
    engine = Engine()
    r = random.Random()

    for i in range(10):
        img = engine.genClassData(r.random()+0.5,           # CubeRho
                            r.random()*np.pi/3-np.pi/6,         # CubeTeta
                            r.random()*2*np.pi,                 # CubeRotZ
                            (r.random(),r.random(),r.random()), # RGB
                            r.random()*1.5+0.5)['X'][0]
        plt.figure()
        plt.imshow(img, interpolation='nearest')
    plt.show()

