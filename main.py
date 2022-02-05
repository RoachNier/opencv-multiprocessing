import time
import cv2
import numpy as np
import multiprocessing as mp
from process.detector import get_points_from_cnts, detector
from process.drawer import drawer

from config import (NUM_OF_FRAMES_TO_STACK,
                    lk_params,
                    colors,
                    video_file_path)

import subprocess as sp
import multiprocessing as mp
from os import remove


start = time.time()


# mog background subtraction
fgbg = cv2.createBackgroundSubtractorMOG2(detectShadows=True)

# read frames from video file
cap = cv2.VideoCapture(video_file_path)

# initiate the mog2 to learn the background
for i in range(1, NUM_OF_FRAMES_TO_STACK):
    # Take first NUM_OF_FRAMES_TO_STACK frame and find corners in it
    ret, frame = cap.read()
    old_blur = cv2.GaussianBlur(frame, (21, 21), 0)
    fgmask = fgbg.apply(old_blur, learningRate=1)
    threshold_frame = cv2.dilate(fgmask, None, iterations=2)
    cnts = cv2.findContours(threshold_frame, cv2.RETR_EXTERNAL,
                            cv2.CHAIN_APPROX_SIMPLE)
    p0, _ = get_points_from_cnts(cnts[0])
    old_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

# Create a mask image for drawing purposes
mask = np.zeros_like(frame)



def process_video_multiprocessing(group_number):
    # Read video file
    cap = cv2.VideoCapture(video_file_path)

    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_jump_unit * group_number)

    # get height, width and frame count of the video
    width, height = (
            int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
            int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        )
    no_of_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))   ##### cv!!!!
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    proc_frames = 0

    # Define the codec and create VideoWriter object
    fourcc = cv2.VideoWriter_fourcc('m', 'p', '4', 'v')
    out = cv2.VideoWriter()
    output_file_name = video_file_path
    out.open("output_{}.mp4".format(group_number), fourcc, fps, (width, height), True)

    try:
        while proc_frames < frame_jump_unit:
            success, frame = cap.read()
            if success == False:
                break

            img = frame

            frame_blur = cv2.GaussianBlur(frame, (21, 21), 0)
            fgmask = fgbg.apply(frame_blur, learningRate=0.2)
            threshold_frame = cv2.dilate(fgmask, None, iterations=2)
            cnts = cv2.findContours(threshold_frame, cv2.RETR_EXTERNAL,
                                    cv2.CHAIN_APPROX_SIMPLE)

            p0, bboxs = get_points_from_cnts(cnts[0])

            if not p0.size:
                continue

            frame_gray, good_new, good_old = detector(frame, old_gray, p0, **lk_params)
            mask, frame = drawer(good_new, good_old, colors, mask, frame, bboxs)

            img = cv2.add(frame, mask)

            cv2.imshow('frame', img)
            k = cv2.waitKey(30) & 0xff
            if k == 27:
                break

            # Now update the previous frame
            old_gray = frame_gray.copy()

            # write the frame
            out.write(img)

            proc_frames += 1    
    except:
        # Release resources
        cap.release()
        out.release()
    


def combine_output_files(num_processes):
    # Create a list of output files and store the file names in a txt file
    list_of_output_files = ["output_{}.mp4".format(i) for i in range(num_processes)]
    with open("list_of_output_files.txt", "w") as f:
        for t in list_of_output_files:
            f.write("file {} \n".format(t))

    # use ffmpeg to combine the video output files
    ffmpeg_cmd = "ffmpeg -y -loglevel error -f concat -safe 0 -i list_of_output_files.txt -vcodec copy " + video_file_path
    sp.Popen(ffmpeg_cmd, shell=True).wait()

    # Remove the temperory output files
    for f in list_of_output_files:
        remove(f)
    remove("list_of_output_files.txt")


def multi_process():
    print("Video processing using {} processes...".format(num_processes))
    start_time = time.time()

    # Paralle the execution of a function across multiple input values
    p = mp.Pool(num_processes)
    p.map(process_video_multiprocessing, range(num_processes))
    print(type(p))

    combine_output_files(num_processes)

    end_time = time.time()

    total_processing_time = end_time - start_time
    print("Time taken: {}".format(total_processing_time))
    print("FPS : {}".format(frame_count/total_processing_time))


def get_video_frame_details(video_file_path):
    cap = cv2.VideoCapture(video_file_path)
    width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
    height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    return width, height, frame_count 


file_name = video_file_path
output_file_name = "output.mp4"
width, height, frame_count = get_video_frame_details(file_name)
print("Video frame count = {}".format(frame_count))
print("Width = {}, Height = {}".format(width, height))

num_processes = mp.cpu_count()
# num_processes = 1

print("Number of CPU: " + str(num_processes))
frame_jump_unit =  frame_count// num_processes

if __name__ == '__main__':
    multi_process()

finish = time.time()
print(f'time spent: {finish - start} s.')  # time spent: 136.53749299049377 s. -> basic solution
