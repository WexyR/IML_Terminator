import pybullet as p

# Should be done only once
p.connect(p.DIRECT)

from engine.utils import euler_to_quaternion

class Cube:
    def __init__(self, position, rotEuler, rgb=(0,0,0), collision=False, size=1.0):
        real_size = size*0.125
        self.visual_id = p.createVisualShape(p.GEOM_BOX,
                                          rgbaColor=list(rgb)+[1],
                                          halfExtents=[real_size, real_size, real_size])
        if collision:
            self.collision_id = p.createCollisionShape(p.GEOM_BOX,
                                          halfExtents=[real_size, real_size, real_size])
            self.body = p.createMultiBody(baseMass=0,
                                          baseCollisionShapeIndex=self.collision_id,
                                          baseVisualShapeIndex=self.visual_id,
                                          basePosition=position,
                                          baseOrientation=euler_to_quaternion(*rotEuler))
        else:
            self.body = p.createMultiBody(baseMass=0,
                                          baseVisualShapeIndex=self.visual_id,
                                          basePosition=position,
                                          baseOrientation=euler_to_quaternion(*rotEuler))
    def remove(self):
        p.removeBody(self.body)
        del self