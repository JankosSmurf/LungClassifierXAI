import numpy as np

class PaletteRedBlue:
    def __init__(self):
        pass

    def palette(self, value):
        if value < 0.5:
            return np.array([0, 0, 1 - (2 * value)])
        if value == 0.5:
            return np.array([0, 0, 0])
        if value > 0.5:
            return np.array([2 * (value - 0.5), 0, 0])

    def __call__(self, numpy_image):
        shape = numpy_image.shape
        rgb_image = np.zeros([shape[0], shape[1], 3])
        for i in range(shape[0]):
            for j in range(shape[1]):
                rgb_image[i][j] = self.palette(numpy_image[i][j])
        return rgb_image
