import cv2 as cv
import numpy as np

from config import (NUM_OF_FRAMES_TO_STACK,
                    lk_params,
                    colors,
                    video_file_path)


def streamer(video_file_path):
    """Get a stream from the capture"""
    cap = cv.VideoCapture(video_file_path)
    
    while True:
        ret, frame = cap.read()
        # here we can add some processing of the frame
        cv.imshow('frame', frame)
        k = cv.waitKey(30) & 0xff
        if k == 27 or ret == False:
            break
    cv.destroyAllWindows()
    cap.release()


# streamer(video_file_path)
