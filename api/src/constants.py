import numpy as np

DETECTION_IMAGE_W = 512
DETECTION_IMAGE_H = 512
PIXEL_MAX_VALUE = 255

RECOGNIZER_IMAGE_W = 128
RECOGNIZER_IMAGE_H = 32

IMG_C = 3

DETECTION_IMG_CONFIGURATION = (2, 0, 1)
RECOGNIZER_IMG_CONFIGURATION = (0, 3, 1, 2)

PLATE_RECT = np.array([
    [0, 0],
    [0, 32],
    [127, 0],
    [127, 31]], dtype='float32')

PLATE_SQUARE = np.array([
    [0, 0],
    [0, 63],
    [63, 0],
    [63, 63]], dtype='float32')
