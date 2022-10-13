import numpy as np
import cv2
from constants import DETECTION_IMAGE_H, DETECTION_IMAGE_W
import constants
from PIL import Image


def prepare_for_detector(image_orig):
    s = max(image_orig.shape[0:2])
    f = np.zeros((s, s, 3), np.uint8)
    # ax, ay = (s - image_orig.shape[1]) // 2, (s - image_orig.shape[0]) // 2
    # f[ay:image_orig.shape[0] + ay, ax:ax + image_orig.shape[1]] = image_orig
    # resized_image = cv2.resize(f.copy(), (DETECTION_IMAGE_W, DETECTION_IMAGE_H))
    resized_image = cv2.resize(image_orig, (DETECTION_IMAGE_W, DETECTION_IMAGE_H))
    image_orig = resized_image.copy()

    resized_image = resized_image.transpose((2, 0, 1))
    resized_image = 2 * (resized_image / 255.0 - 0.5)
    x = resized_image.astype(np.float32)
    x = np.ascontiguousarray(x)

    return image_orig, x

def bbox_iou_np(box1, box2, x1y1x2y2=True):
    if not x1y1x2y2:

        b1_x1, b1_x2 = box1[:, 0] - box1[:, 2] / 2, box1[:, 0] + box1[:, 2] / 2
        b1_y1, b1_y2 = box1[:, 1] - box1[:, 3] / 2, box1[:, 1] + box1[:, 3] / 2
        b2_x1, b2_x2 = box2[:, 0] - box2[:, 2] / 2, box2[:, 0] + box2[:, 2] / 2
        b2_y1, b2_y2 = box2[:, 1] - box2[:, 3] / 2, box2[:, 1] + box2[:, 3] / 2
    else:
        b1_x1, b1_y1, b1_x2, b1_y2 = box1[:, 0], box1[:, 1], box1[:, 2], box1[:, 3]
        b2_x1, b2_y1, b2_x2, b2_y2 = box2[:, 0], box2[:, 1], box2[:, 2], box2[:, 3]

    inter_rect_x1 = np.maximum(b1_x1, b2_x1)
    inter_rect_y1 = np.maximum(b1_y1, b2_y1)
    inter_rect_x2 = np.minimum(b1_x2, b2_x2)
    inter_rect_y2 = np.minimum(b1_y2, b2_y2)

    inter_area = np.clip(inter_rect_x2 - inter_rect_x1 + 1, 0, None) * \
                 np.clip(inter_rect_y2 - inter_rect_y1 + 1, 0, None)

    b1_area = (b1_x2 - b1_x1 + 1) * (b1_y2 - b1_y1 + 1)
    b2_area = (b2_x2 - b2_x1 + 1) * (b2_y2 - b2_y1 + 1)

    iou = inter_area / (b1_area + b2_area - inter_area)

    return iou


def nms_np(predictions, conf_thres=0.2, nms_thres=0.2, include_conf=False):
    filter_mask = (predictions[:, -1] >= conf_thres)
    predictions = predictions[filter_mask]

    if len(predictions) == 0:
        return np.array([])

    output = []

    while len(predictions) > 0:
        max_index = np.argmax(predictions[:, -1])

        if include_conf:
            output.append(predictions[max_index])
        else:
            output.append(predictions[max_index, :-1])

        ious = bbox_iou_np(np.array([predictions[max_index, :-1]]), predictions[:, :-1], x1y1x2y2=False)

        predictions = predictions[ious < nms_thres]

    return np.stack(output)


def preprocess_image_recognizer(img, box):
    is_squared = False
    plate_imgs = []
    ratio = abs((box[4, 0] - box[3, 0]) / (box[3, 1] - box[2, 1]))
    if 2.6 >= ratio >= 0.8:
        plate_img = cv2.warpPerspective(img, cv2.getPerspectiveTransform(box[2:], constants.PLATE_SQUARE),
                                        (constants.RECOGNIZER_IMAGE_W // 2, constants.RECOGNIZER_IMAGE_H * 2))

        padding = np.ones((constants.RECOGNIZER_IMAGE_H,
                           constants.RECOGNIZER_IMAGE_W // 2, 3), dtype=np.uint8) * constants.PIXEL_MAX_VALUE

        first_half = np.concatenate((plate_img[:constants.RECOGNIZER_IMAGE_H], padding), axis=1).astype(
            np.uint8)
        second_half = np.concatenate((plate_img[constants.RECOGNIZER_IMAGE_H:], padding), axis=1).astype(
            np.uint8)
        plate_imgs.append(first_half)
        plate_imgs.append(second_half)
        is_squared = True
    else:
        plate_img = cv2.warpPerspective(img, cv2.getPerspectiveTransform(box[2:], constants.PLATE_RECT),
                                        (constants.RECOGNIZER_IMAGE_W, constants.RECOGNIZER_IMAGE_H))
        plate_imgs.append(plate_img)
        is_squared = False
    return np.ascontiguousarray(
        np.stack(plate_imgs).astype(np.float32).transpose(
                constants.RECOGNIZER_IMG_CONFIGURATION) / constants.PIXEL_MAX_VALUE), is_squared

