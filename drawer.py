import cv2
import numpy as np


def drawer(good_new, good_old, colors, mask, frame, bboxs):
    # draw the tracks
    for i, (new, old) in enumerate(zip(good_new, good_old)):
        a, b = new.ravel()
        c, d = old.ravel()
        # mask = cv2.line(mask, (a, b), (c, d), colors[i].tolist(), 2)
        # frame = cv2.circle(frame, (a, b), 7, colors[i].tolist(), -1)
        mask = cv2.line(mask, (int(a), int(b)), (int(c), int(d)), colors[i].tolist(), 2)
        frame = cv2.circle(frame, (int(a), int(b)), 7, colors[i].tolist(), -1)
    
    # draw the bounding box
    for (x, y, w, h) in bboxs:
        frame = cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

    return mask, frame
