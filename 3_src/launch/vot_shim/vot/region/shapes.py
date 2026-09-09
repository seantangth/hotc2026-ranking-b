import numpy as np


class Rectangle:
    def __init__(self, x, y, width, height):
        self.x, self.y, self.width, self.height = float(x), float(y), float(width), float(height)


class Mask:
    def __init__(self, m):
        self._m = np.asarray(m, dtype=bool)

    def convert(self, kind):
        ys, xs = np.nonzero(self._m)
        if len(xs) == 0:
            return Rectangle(0.0, 0.0, 0.0, 0.0)
        return Rectangle(xs.min(), ys.min(), xs.max() - xs.min() + 1, ys.max() - ys.min() + 1)
