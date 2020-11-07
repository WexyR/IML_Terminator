from enum import Enum
import numpy as np

class Colors(Enum):
    MAROON = [0.5, 0.0, 0.0]
    RED = [1.0, 0.0, 0.0]
    ORANGE = [1.0, 0.65, 0.0]
    YELLOW = [1.0, 1.0, 0.0]
    OLIVE = [0.5, 0.5, 0.0]
    GREEN = [0.0, 0.5, 0.0]
    PURPLE = [0.5, 0.0, 0.5]
    FUSHIA = [1.0, 0.0, 1.0]
    LIME = [0.0, 1.0, 0.0]
    TEAL = [0.0, 0.5, 0.5]
    AQUA = [0.0, 1.0, 1.0]
    BLUE = [0.0, 0.0, 1.0]
    NAVY = [0.0, 0.0, 0.5]
    BLACK = [0.0, 0.0, 0.0]
    GRAY = [0.5, 0.5, 0.5]
    SILVER = [0.75, 0.75, 0.75]
    WHITE = [1.0, 1.0, 1.0]

    @staticmethod
    def classifiate(rgb):
        selected = -1
        dist = 100000
        tmp = 0
        index = -1
        for entry in Colors:
            index+=1
            tmp = np.sum((entry.value[i] - rgb[i]) ** 2 for i in range(3))
            if tmp < dist:
                selected = index
                dist = tmp
        return selected