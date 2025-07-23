import cv2 as cv
import numpy as np
import math
import time
from blimp_vision_msgs.msg import Detection

######### Functions ########

# True if area > min area and percent of enclosing circle filled > min percent filled
def isValidArea(contour):
    cont_area = cv.contourArea(contour) 
    (x,y), radius = cv.minEnclosingCircle(contour)
    encl_area = math.pi * radius * radius
    if (cont_area/encl_area)*100 > min_percent_filled and cont_area >= min_area:
        return True
    return False

######## Define prarmeters ########

# Green HSV
green_uh = 76
green_lh = 39
green_us = 153
green_ls = 78
green_uv = 219
green_lv = 58

# Purple HSV
purple_uh = 133
purple_lh = 106
purple_us = 189
purple_ls = 30
purple_uv = 180
purple_lv = 62

# Filters
min_percent_filled = 60
min_area = 250
include_green = True
include_purple = True

# Tracking
use_kalman = True
use_optical_flow = True
use_lock = True

R_pos = 4 ** 2
R_v = 5 ** 2
Q_pos = 0.08 ** 2
Q_v = 0.08 ** 2
v_scale = .8
lock_radius = 2

# Debug
show_time = False

# Inital variables
x = -1
y = -1
prev_x = -1
prev_y = -1
x_est = -1
y_est = -1
v_x_s = 0
v_y_s = 0
p0 = None
start_time = None
radius = -1

# Kalman filter
kalman_initialized = False
no_detect_count = 0

# Lock
sim_detect_count = 0
lock_on = False

# Optical flow
# ShiTomasi corner detection
feature_params = dict( maxCorners = 100,
                       qualityLevel = 0.3,
                       minDistance = 7,
                       blockSize = 7 )
# Lucas kanade optical flow
lk_params = dict( winSize  = (15, 15),
                  maxLevel = 2,
                  criteria = (cv.TERM_CRITERIA_EPS | cv.TERM_CRITERIA_COUNT, 10, 0.03))
old_gray = None



def contour_find_ball(frame):

    ######### Blob detection #########

    if show_time:
        start_time = time.time()

    # Convert to HSV
    hsv_frame = cv.cvtColor(frame, cv.COLOR_BGR2HSV)

    # Apply adaptive mask to HSV frame if lock is on
    if lock_on and use_lock:
        height, width, n = frame.shape
        prev_detection_mask = np.zeros((height, width), dtype=np.uint8)
        cv.circle(prev_detection_mask, (int(x_est),int(y_est)), lock_radius * 2, 255, -1)
        hsv_frame = cv.bitwise_and(hsv_frame, hsv_frame, mask = prev_detection_mask)

    # If ballon is close increase value
    if radius > 40:
        purple_uv = 250
        green_uv = 250

    # Create masks
    g_upper = np.array([green_uh,green_us,green_uv])
    g_lower = np.array([green_lh,green_ls,green_lv])
    p_upper = np.array([purple_uh,purple_us,purple_uv])
    p_lower = np.array([purple_lh,purple_ls,purple_lv])

    # Extend bounds if lock is on
    if lock_on:
        g_upper += 5
        g_lower -= 5
        p_upper += 5
        p_lower -= 5
        
    p_mask = cv.inRange(hsv_frame, p_lower, p_upper)
    g_mask = cv.inRange(hsv_frame, g_lower, g_upper)

    # Filter out green or purple if turned off
    if not include_green:
        g_mask = np.zeros_like(g_mask)
    if not include_purple:
        p_mask = np.zeros_like(p_mask)

    # Combine green and purple masks
    combined_mask = g_mask + p_mask

    # Erode and dilate mask
    if lock_on:
        erode_kernel = np.ones((3,3),np.uint8)
        dilate_kernel = np.ones((13,13),np.uint8)
    else:
        erode_kernel = np.ones((4,4),np.uint8)
        dilate_kernel = np.ones((10,10),np.uint8)

    combined_mask = cv.erode(combined_mask, erode_kernel)
    combined_mask = cv.dilate(combined_mask, dilate_kernel)

    # Create masked frame
    masked_frame = cv.bitwise_and(frame, frame, mask = combined_mask)

    # Create and draw contours on masked frame
    ret, thresh = cv.threshold(combined_mask, 127, 255, 0)
    contours, hierarchy = cv.findContours(thresh, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)

    # Find best contour based on max area and filters
    max_area = 0
    best_i = -1
    for i in range(len(contours)):
        cont_area = cv.contourArea(contours[i]) 
        if cont_area > max_area and isValidArea(contours[i]):
            max_area = cont_area
            best_i = i

    if best_i > -1:
        (x,y), radius = cv.minEnclosingCircle(contours[best_i])
        no_detect_count = 0
    else:
        x = -1
        y = -1
        no_detect_count += 1

    # Print blob detection time
    if show_time and start_time is not None:
        end_time = time.time()
        elapsed_time1 = end_time - start_time
        print('\nBlob detection: ', elapsed_time1)

        start_time = time.time()

    ######## Optical flow ########

    if use_optical_flow:
        if x > -1:
            if old_gray is None:
                old_gray = cv.cvtColor(frame,cv.COLOR_BGR2GRAY)

            height, width, n = frame.shape
            optical_flow_mask = np.zeros((height, width), dtype=np.uint8)
            cv.circle(optical_flow_mask, (int(x),int(y)), int(radius), 255, -1)
            p0 = cv.goodFeaturesToTrack(old_gray, mask = optical_flow_mask, **feature_params)

        if p0 is not None:
            frame_sparse = frame.copy()
            frame_gray = cv.cvtColor(frame_sparse, cv.COLOR_BGR2GRAY)
            p1, st, err = cv.calcOpticalFlowPyrLK(old_gray, frame_gray, p0, None, **lk_params)

            # Select good points
            if p1 is not None:
                good_new = p1[st==1]
                good_old = p0[st==1]

            # Calc velocity
            feat_count = 0
            v_x_s = 0
            v_y_s = 0
            for i, (new, old) in enumerate(zip(good_new, good_old)):
                a, b = new.ravel()
                c, d = old.ravel()
                v_x_s += c - a
                v_y_s += b - d
                feat_count += 1

            if not feat_count == 0:
                v_x_s /= feat_count
                v_y_s /= feat_count

                v_x_s *= v_scale
                v_y_s *= v_scale

                if abs(v_x_s) > 5:
                    v_x_s = 0

                if abs(v_y_s) > 5:
                    v_y_s = 0

            # Now update the previous frame and previous points
            old_gray = frame_gray.copy()
            p0 = good_new.reshape(-1, 1, 2)

    # Print optical flow time
    if show_time and start_time is not None:
        end_time = time.time()
        elapsed_time2 = end_time - start_time
        print('Optical flow: ', elapsed_time2)

        start_time = time.time()

    ######## Kalman filter ########

    if kalman_initialized and use_kalman:
        # Estimate
        Q = np.matrix([[Q_pos,0,0,0],[0,Q_pos,0,0],[0,0,Q_v,0],[0,0,0,Q_v]])
        dt = 1
        F_t = np.matrix([[1,0,dt,0], [0,1,0,dt], [0,0,1,0], [0,0,0,1]])

        # Predict
        x_hat = F_t @ x_prev
        P_t = F_t @ P_prev @ F_t.getT() + Q

        # Measurement
        if use_optical_flow:
            v_x = v_x_s
            v_y = v_y_s
        elif prev_x > -1:
            v_x = x - prev_x
            v_y = y - prev_y

        # Update
        if x > -1:
            z_t = np.matrix([[x],[y],[v_x],[v_y]])
            H_t = np.matrix([[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1]])
            R = np.matrix([[R_pos,0,0,0],[0,R_pos,0,0],[0,0,R_v,0],[0,0,0,R_v]])
            I = np.matrix([[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1]])
            res = z_t - (H_t @ x_prev)
            S = H_t @  P_t @ H_t.getT() + R
            K = P_t @ H_t.getT() @ S.getI()
            x_hat = x_hat + K @ res
            P_t = (I - K @ H_t) @ P_t

        x_est = x_hat[0].item()
        y_est = x_hat[1].item()

        P_prev = P_t
        x_prev = x_hat

        if no_detect_count > 100 or (x > -1 and (abs(x_est - x) > 10 and abs(y_est - y) > 10)):
            kalman_initialized = False
            x_est = -1
            y_est = -1

    elif x > -1:
            x_prev = np.matrix([[x],[y],[0],[0]])
            P_prev = np.matrix([[10,0,0,0],[0,10,0,0],[0,0,10,0],[0,0,0,10]])
            kalman_initialized = True

    # Count for how many frames the detection is in a similar area to prev detection
    if x_est > - 1 and abs(x_est - prev_x_est) < 5 and abs(y_est - prev_y_est) < 5:
        sim_detect_count += 1
    else:
        sim_detect_count = 0
        lock_on = False

    # Enable lock if balloon is in similar place for 50 frames
    if sim_detect_count > 20:
        lock_on = True

    # Update previous values
    prev_x = x
    prev_y = y
    prev_x_est = x_est
    prev_y_est = y_est

    # Time for kalman filter and total time
    if show_time and start_time is not None:
        end_time = time.time()
        elapsed_time3 = end_time - start_time
        print('Kalman: ', elapsed_time3)
        total_time = elapsed_time1 + elapsed_time2 + elapsed_time3
        print('Total: ', total_time)
        print('fps: ', 1/total_time)

    # Streaming
    # cv.circle(frame, (int(x_est),int(y_est)), int(radius), (0, 0, 255), 3)


    ######## Return detection ########
    diameter = radius * 2

    detection_msg = Detection()
    detection_msg.class_id = 1
    detection_msg.obj_class = "Shape"
    detection_msg.bbox[0] = x_est
    detection_msg.bbox[1] = y_est
    detection_msg.bbox[2] = diameter
    detection_msg.bbox[3] = diameter
    detection_msg.depth = -1.0
    detection_msg.confidence = -1.0
    detection_msg.track_id = -1

    return detection_msg