from enum import Enum, unique
import numpy as np

class _Colors(Enum):
    """
    deprecated
    """
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
        """

        Parameters
        ----------
        rgb a size 3 int tuple describing a color

        Returns the class id of this color determined using least square distance
        -------

        """

        selected = -1
        dist = 100000
        tmp = 0
        index = -1
        for entry in _Colors:
            index+=1
            tmp = np.sum((entry.value[i] - rgb[i]) ** 2 for i in range(3))
            if tmp < dist:
                selected = index
                dist = tmp
        return selected

@unique
class Colors(Enum):
    BLACK=0
    RED=1
    GREEN=2
    YELLOW=3
    BLUE=4
    MAGENTA=5
    CYAN=6
    WHITE=7

    def next_random_colors(N=1, rgb_factor=1):
        """
        Parameters
        ----------
        N is the size of the output vector
        rgb_factor affects the rgb format. 255 means a 0-255 color format

        Returns a tuple (tags, colors), tags being the index of the class, and colors a generated rgb
        -------
        """
        assert N>0
        val = np.clip(np.random.normal(loc=0, scale=0.1581, size=3*N), -0.5, 0.5)
        res = np.where(val<0, 1+val, val)
        tag = np.packbits(np.array_split(np.rint(res).astype(np.uint8), N), axis=1, bitorder="little")
        res *= rgb_factor
        return tag.transpose()[0], np.array_split(res, N)