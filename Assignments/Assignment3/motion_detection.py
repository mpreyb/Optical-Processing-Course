"""
    This code aims to detect object motion.
    
    File name: motion_detection.py
    Author: Maria Paula Rey
    Date last modified: 09/11/2021
    Python Version: 3.8
    
    This code is intended for command propmt execution.
"""

# Importing necessary packages
from imutils.video import VideoStream
import numpy as np
import argparse
import imutils
import cv2
import os
import sys

# This is where our videos are stored
search_paths = ('.' , './videos')
videos = ('bee','fly', 'ping')

# Construct the argument parser and parse the arguments
# To execute open command prompt and type "python motiondetect.py -v [video name] + any needed command :
ap = argparse.ArgumentParser()
ap.add_argument("-v", "--video", help="path to the video file")
ap.add_argument("-a", "--min-area", type=int, default=1000, help="minimum area size")
ap.add_argument('-f', '--fps', type=float, default=30, help="Desired FPS")
ap.add_argument("-t", "--min-threshold", type=int, default=25, help="minimum threshold")
ap.add_argument("-r", "--resize", action="store_true", help="resize video")
ap.add_argument("-R", "--record", action="store_true", help="save the resulting videos")
args = vars(ap.parse_args())


# --------------------------------------Finding video-----------------------------------------
try:
 # If input was a number
  pos = int(args['video']) % len(videos)  # Assigns position to selected video
  vid = videos[pos]                       # This is our selected video from list "videos"
 
# If input was of the form video.avi   
except ValueError:
        res = args['video'].split('.')          # Obtain video name and extension
        if len(res) == 2:
            vid = res[0]
            video_name = args['video']
        else:
            vid = res[0]
            video_name = f"{vid}.avi"
            
video_name = f"{vid}.avi"                  # Assigns .avi extension to our file name 
found = False                               # Video has not been found
path = ''

for p in search_paths:
    path = f"{p}/{video_name}"
    if os.path.isfile(path):               # Returns true if path is an existing regular file.
     found = True                       # Video has been found
     break
        
if not found:
        print(f"Error: Video file '{video_name}' not found in the following search directories:\n{search_paths}", file=sys.stderr)
        sys.exit(1)

vs = cv2.VideoCapture(path)
    
#-----------------------------------------------------------------------------------------------

kernel_size = (21, 21)
firstFrame = None

# Initializing videos we need. They are empty at first
vid_in = None
vid_out = None
vid_out_delta = None
vid_out_thr = None

try:
    # Loop over the frames of the video
    while True:
        
        # Grab the current frame and initialize the occupied/unoccupied
        frame = vs.read()
        frame = frame if args.get("video", None) is None else frame[1]   #If nothing has changed, i.e nothing is moving
        text = "Static"

        # If the frame could not be obtained, then the video has ended
        if frame is None:
            break

        if args["resize"]:                                     # Resize the frame
            frame = imutils.resize(frame, width=500)      
        frame_orig = frame
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)         # Grayscale
        gray = cv2.GaussianBlur(gray, kernel_size, 0)          # Gaussian Blur

        # If the first frame is None, initialize it
        if firstFrame is None:
            firstFrame = gray                                  # Grayscale
            frame_height, frame_width = firstFrame.shape[:2]
            
            if args["record"]:                                 # If we want to save resulting .avi files
                if args["resize"]:
                    vid_in = cv2.VideoWriter(f"motion_{vid}_in.avi", cv2.VideoWriter_fourcc('M','J','P','G'), args['fps'], (frame_width,frame_height))
                vid_out = cv2.VideoWriter(f"motion_{vid}_out.avi", cv2.VideoWriter_fourcc('M','J','P','G'), args['fps'], (frame_width,frame_height))
                vid_out_delta = cv2.VideoWriter(f"motion_{vid}_delta.avi", cv2.VideoWriter_fourcc('M','J','P','G'), args['fps'], (frame_width,frame_height))
                vid_out_thr = cv2.VideoWriter(f"motion_{vid}_threshold.avi", cv2.VideoWriter_fourcc('M','J','P','G'), args['fps'], (frame_width,frame_height))
            continue

        # Absolute difference between the current frame and previous frame.
        frameDelta = cv2.absdiff(firstFrame, gray)
        
        # Thresholding diffence (binary representation of difference between current and previous frame)
        thresh = cv2.threshold(frameDelta, args["min_threshold"], 255, cv2.THRESH_BINARY)[1]

        # Dilate the thresholded image to fill in holes
        thresh = cv2.dilate(thresh, None, iterations=5)
        
        #Find contours on thresholded image
        cnts = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE)
        cnts = imutils.grab_contours(cnts)

        # Looping over the contours
        for c in cnts:
            
            # If the contour is too small, ignore it
            if cv2.contourArea(c) < args["min_area"]:
                continue

            # Bounding box for the contour, draw it on the frame, and update the text
            (x, y, w, h) = cv2.boundingRect(c)
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            text = "Motion detected"

        # Draw the text
        cv2.putText(frame, f"Scene: {text}", (10, 20),
            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

        # Show the frame and record if the user requests it
        if args["record"]:
            if args["resize"]:
                vid_in.write(frame_orig)    # Saving resized original video
            vid_out.write(frame)
            vid_out_delta.write(np.repeat(frameDelta[..., None], 3, axis=2))
            vid_out_thr.write(np.repeat(thresh[..., None], 3, axis=2))
        else:
            cv2.imshow("Video", frame)
            cv2.imshow("Thresholding + dilation", thresh)
            cv2.imshow("Delta", frameDelta)
            
        key = cv2.waitKey(1) & 0xFF

        # If the `E` key is pressed, break from the loop
        if key == ord("E"):
            break
        
        firstFrame = gray
    
except Exception as ex:
    print('An error ocurred:')
    print(ex)
finally:
    if args["record"]:
        if args["resize"]:
            vid_in.release()
        vid_out.release()
        vid_out_delta.release()
        vid_out_thr.release()

# Clean up the camera and close any open windows
vs.stop() if args.get("video", None) is None else vs.release()
cv2.destroyAllWindows()