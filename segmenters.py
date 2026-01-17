from skimage.filters import sobel
import numpy as np
from skimage.color import rgb2gray
from skimage.segmentation import felzenszwalb, slic, quickshift, watershed


### Wrapper for skimage segmenters

class MySegmenter:
    def __init__(self, type, param1, param2, param3):
        self.type = type
        self.param1 = param1
        self.param2 = param2
        self.param3 = param3

    def __call__(self, image):
        if self.type == 'felzenszwalb':
            # Example values 200, 0.5, 50
            segments = felzenszwalb(image, scale=self.param1, sigma=self.param2, min_size=self.param3)
        elif self.type == 'slic':
            # Example values 50, 10, 1
            segments = slic(image, n_segments=self.param1, compactness=self.param2, sigma=self.param3)
        elif self.type == 'quickshift':
            # Example values 7, 100, 0.5
            segments = quickshift(image, kernel_size=self.param1, max_dist=self.param2, ratio=self.param3)
        elif self.type == 'watershed':
            # Example values 50, 0.001, -
            gradient = sobel(rgb2gray(image))
            segments = watershed(gradient, markers=self.param1, compactness=self.param2)
        else:
            pass
        return segments



### Additional mask to cover background

class LungSegmenter:
    def __init__(self, main_segmenter):
        self.mask = np.load("other/segmentation_mask.npy")
        self.segmenter = main_segmenter
    def __call__(self, image):
        segmented = self.segmenter(image)
        background_class = len(np.unique(segmented))
        for i in range(segmented.shape[0]):
            for j in range(segmented.shape[1]):
                if self.mask[i, j] == 0:
                    segmented[i, j] = background_class
        for i in range(np.min(segmented)):
            for k in range(64):
                for l in range(64):
                    if segmented[k, l] == np.unique(segmented)[-1]:
                        segmented[k, l] = i
        for i in range(1,len(np.unique(segmented))):
            if np.unique(segmented)[i] - np.unique(segmented)[i-1] != 1:
                difference = np.unique(segmented)[i] - np.unique(segmented)[i-1]
                for k in range(64):
                    for l in range(64):
                        if segmented[k, l] > np.unique(segmented)[i-1]:
                            segmented[k, l] -= difference -1
        return segmented
