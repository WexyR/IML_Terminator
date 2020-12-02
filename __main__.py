import threading
import time

import resources
from classifier import Classifier
from engine import Engine, Cube, Colors
import pybullet as p
import numpy as np
import random


if __name__ == "__main__":
    engine = Engine(gui=True)

    classifier = Classifier()
    classifier.load(resources.CNNS)

    rgb_ids = (p.addUserDebugParameter("red", 0, 1, 0.5),
               p.addUserDebugParameter("green", 0, 1, 0.5),
               p.addUserDebugParameter("blue", 0, 1, 0.5))
    theta, rho, angle = (p.addUserDebugParameter("theta", 0, 360, 0),
                  p.addUserDebugParameter("rho", 0.5, 1.5, 1),
                         p.addUserDebugParameter("angle", 0, 360, 0))
    cube_size = p.addUserDebugParameter("size", 0.5, 2.0, 1.0)
    last_cube_update = time.time()


    def update_debug_parameters():
        global last_cube_update
        cube = None
        last_pos, last_angle, last_size = (None, None, None)
        while p.isConnected():
            rgb = [p.readUserDebugParameter(id) for id in rgb_ids] + [1]
            _theta, _rho, _angle = (p.readUserDebugParameter(theta)/180.0*np.pi,
                                    p.readUserDebugParameter(rho),
                                    p.readUserDebugParameter(angle)/180.0*np.pi)
            _cube_size = p.readUserDebugParameter(cube_size)*0.125
            pos = (np.cos(_theta)*_rho, np.sin(_theta)*_rho, _cube_size/2.0)
            if last_pos!=pos or last_angle!=_angle or last_size!=_cube_size:
                if cube is not None:
                    last_cube_removal = time.time()
                    cube.remove()
                cube = Cube(pos,
                            (_angle, 0, 0),
                            rgb,
                            _cube_size)
                last_pos = pos
                last_angle = _angle
                last_size = _cube_size
                continue
            elif cube is not None:
                p.changeVisualShape(cube.body, -1, rgbaColor=rgb)
            time.sleep(0.2)

    def pepper_track_object():
        found_left, found_right, found_mid = False, False, False
        while p.isConnected():
            if time.time() - last_cube_update < 2:
                found_left, found_right, found_mid = False, False, False
                time.sleep(0.5)
            right, mid, left = (engine._agent.getRightLaserValue(),engine._agent.getFrontLaserValue(),engine._agent.getLeftLaserValue())

            found_mid = False
            for j in mid:
                if j < 2.8:
                    found_mid = True
                    found_left = False
                    found_right = False
                    break
            if not found_mid:
                for j in right:
                    if j < 2.8:
                        found_right = True
                        found_left = False
                    break
                for j in left:
                    if j < 2.8:
                        found_left = True
                        found_right = False
                    break
                if not found_right:
                    if found_left:
                        engine._agent.move(0, 0, np.pi / 2.0)
                    else:
                        if random.random()>0.5:
                            found_left = True
                        else:
                            found_right = True
                else:
                    engine._agent.move(0, 0, -np.pi / 2.0)
            else:
                tmp=[]
                for j in range(len(mid)):
                    if(mid[j] != 3.0):
                        tmp+=[j]
                center = np.mean(tmp)
                dif = (center-len(mid)/2.0)/len(mid)*2.0
                engine._agent.move(0, 0, -np.pi / 2.0 * dif)

    t1 = threading.Thread(target=update_debug_parameters)
    t1.start()
    t2 = threading.Thread(target=pepper_track_object)
    t2.start()

    color_debug = {"accuracy":0, "class":None, "overlay_id":None}
    while p.isConnected():
        X = engine.screenshot()
        predicate = classifier.predict([np.array([entry]) for entry in X])
        index = np.argmax(predicate)
        color_debug["accuracy"] = predicate.T[index][0]
        color_debug["class"] = str(list(Colors)[index]).split('.')[1]

        if color_debug["overlay_id"] is not None:
            p.removeUserDebugItem(color_debug["overlay_id"])
        color_debug["overlay_id"] = p.addUserDebugText(f"{color_debug['class']} @ {color_debug['accuracy']*100:.1f}%",
                                        (0, -0.7, 1.5),
                                        textSize=0.3,
                                        textColorRGB=(0, 0, 0),
                                        textOrientation=p.getQuaternionFromEuler((np.pi/2, 0, np.pi/2)),
                                        parentObjectUniqueId=engine._agent.getRobotModel()
                                        )

        print(list(Colors)[np.argmax(predicate)])
    t1.join()
    t2.join()

    engine.stop()
