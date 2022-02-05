import cv2
import numpy as np


def get_points_from_cnts(cnts):
    points = []
    bboxs = []
    for c in cnts:
        # print(cv2.contourArea(c))   # uncomment this for testing
        # if the contour is too small or too big, ignore it
        if cv2.contourArea(c) < 500 or cv2.contourArea(c) > 50000:
            continue

        # compute the bounding box for the contour, draw it on the frame,
        (x, y, w, h) = cv2.boundingRect(c)
        center_x, center_y = (x + x + w) / 2, (y + y + h) / 2
        points.append([[center_x, center_y]])  # insert the centroid
        bboxs.append((x, y, w, h))
        
        # print(f'points for c_{c}:{[center_x, center_y]}')
        # print(f'bbox for c_{c}:{(x,y,w,h)}')

    return np.array(points).astype("float32"), bboxs


def detector(frame, old_gray, p0, **lk_params):
    # calculate optical flow
    frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    p1, st, err = cv2.calcOpticalFlowPyrLK(old_gray, frame_gray, p0, None, **lk_params)
    # print(f'optical flow: (p1: {p1}, st: {st}, err: {err}')

    # Select good points
    if p1 is not None:
        good_new = p1[st==1]
        good_old = p0[st==1]

    return frame_gray, good_new, good_old
    