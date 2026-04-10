import cv2 as cv
import numpy as np

width = 1280
out_width = 640
height = 480
fps = 15

cap = cv.VideoCapture(0, cv.CAP_V4L2)
fourcc = cv.VideoWriter_fourcc(*'XVID')
out = cv.VideoWriter('van_leer_balloon.avi', fourcc, fps, (out_width, height))

cap.set(cv.CAP_PROP_FOURCC, cv.VideoWriter.fourcc('M', 'J', 'P', 'G')) # Either MJPG or YUYV
cap.set(cv.CAP_PROP_AUTO_WB, 1) # 0=manual, 1=automatic
cap.set(cv.CAP_PROP_AUTO_EXPOSURE, 3) # 1=manual, 3=automatic
#cap.set(cv.CAP_PROP_EXPOSURE, 1251) # good values might be 157, 313, 626, 1251, 2501
cap.set(cv.CAP_PROP_FRAME_WIDTH, width)
cap.set(cv.CAP_PROP_FRAME_HEIGHT, height)
cap.set(cv.CAP_PROP_FPS, fps)

# use v4l2-ctl --list-devices to see which /dev/video# the camera is, then
# use v4l2-ctl -d /dev/video# --list-formats-ext to see available resolution/fps formats for MJPG or YUYV
# Available formats
# MJPG:
#   3840x1080: 60, 30, 25, 20, 15, 10, 5 fps
#   2560x720:  60, 30, 25, 20, 15, 10, 5 fps
#   1600x600:  60, 30, 25, 20, 15, 10, 5 fps
#   1280x480:  60, 30, 25, 20, 15, 10, 5 fps
#   640x240:   60, 30, 25, 20, 15, 10, 5 fps
# YUYV:
#   3840x1080: 1 fps
#   2560x720:  5 fps
#   1600x600:  10, 5 fps
#   1280x480:  15, 10, 5 fps
#   640x240:   60, 30, 25, 20, 15, 10, 5 fps


if not cap.isOpened():
    print("Cannot open camera")
    exit()
while True:
    try:
        ret, frame = cap.read()
        height, width, n = frame.shape

        frame_width = frame.shape[1] // 2
        left_frame = frame[:, :frame_width]
        right_frame = frame[:, frame_width:]

        out.write(left_frame)
    except KeyboardInterrupt:
        print("\nStopped by user (Ctrl+C)")
        break

cap.release()
out.release()
cv.destroyAllWindows()