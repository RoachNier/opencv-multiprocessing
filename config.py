import cv2
import numpy as np

NUM_OF_FRAMES_TO_STACK = 50
video_file_path = 'MOT20-02-raw.mp4'
# Parameters for lucas kanade optical flow
lk_params = dict(winSize=(15, 15),
                 maxLevel=2,
                 criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))

# Create some random colors
colors = np.random.randint(0, 255, (1000, 3))
