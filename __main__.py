import threading
import time

import resources
from classifier import Classifier
from engine import Engine, Cube, Colors
import pybullet as p
import numpy as np


if __name__ == "__main__":
    engine = Engine(gui=True)

    classifier = Classifier()
    classifier.load(resources.CNNS)

    cube = Cube((1, 0, 0),
                (0, 0, 0),
                Colors.next_random_colors()[1][0],
                1.0)
    rgb_ids = [p.addUserDebugParameter("red", 0, 1, 0.5),
               p.addUserDebugParameter("green", 0, 1, 0.5),
               p.addUserDebugParameter("blue", 0, 1, 0.5)]


    def update_rgb():
        while p.isConnected():
            rgb = [p.readUserDebugParameter(id) for id in rgb_ids] + [1]
            p.changeVisualShape(cube.body, -1, rgbaColor=rgb)
    t1 = threading.Thread(target=update_rgb)
    t1.start()

    while p.isConnected():
        t0 = time.time()
        X = engine.screenshot(skip_frame=False)
        data = [np.array([entry]) for entry in X]
        #print(f'screenshot:{time.time()-t0}')
        t0 = time.time()
        predicate = classifier.predict(data)
        #print(f'predict:{time.time()-t0}')
        print(list(Colors)[np.argmax(predicate)])
    t1.join()

    # TODO: have pepper face the cube. Enable the user to move the cube

    engine.stop()