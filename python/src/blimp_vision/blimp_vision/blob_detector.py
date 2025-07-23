import cv2 as cv
import numpy as np
import math
import time
from blimp_vision_msgs.msg import Detection


class BlobDetectorClass:
    def __init__(self):
         ######## Define prarmeters ########
        # Green HSV
        self.green_uh = 76
        self.green_lh = 39
        self.green_us = 153
        self.green_ls = 78
        self.green_uv = 219
        self.green_lv = 58

        # Purple HSV
        self.purple_uh = 133
        self.purple_lh = 106
        self.purple_us = 189
        self.purple_ls = 30
        self.purple_uv = 180
        self.purple_lv = 62

        # Filters
        self.min_percent_filled = 60
        self.min_area = 250
        self.include_green = True
        self.include_purple = True

        # Tracking
        self.use_kalman = True
        self.use_optical_flow = True
        self.use_lock = True

        self.R_pos = 4 ** 2
        self.R_v = 5 ** 2
        self.Q_pos = 0.08 ** 2
        self.Q_v = 0.08 ** 2
        self.v_scale = .8
        self.lock_radius = 2

        # Debug
        self.show_time = False

        # Inital variables
        self.x = -1
        self.y = -1
        self.prev_x = -1
        self.prev_y = -1
        self.x_est = -1
        self.y_est = -1
        self.prev_x_est = -1
        self.prev_y_est = -1
        self.v_x_s = 0
        self.v_y_s = 0
        self.p0 = None
        self.start_time = None
        self.radius = -1

        # Kalman filter
        self.kalman_initialized = False
        self.no_detect_count = 0

        # Lock
        self.sim_detect_count = 0
        self.lock_on = False

        # Optical flow
        # ShiTomasi corner detection
        self.feature_params = dict( maxCorners = 100,
                            qualityLevel = 0.3,
                            minDistance = 7,
                            blockSize = 7 )
        # Lucas kanade optical flow
        self.lk_params = dict( winSize  = (15, 15),
                        maxLevel = 2,
                        criteria = (cv.TERM_CRITERIA_EPS | cv.TERM_CRITERIA_COUNT, 10, 0.03))
        self.old_gray = None

        self.cap = cv.VideoCapture('output.avi')
        


    # True if area > min area and percent of enclosing circle filled > min percent filled
    def isValidArea(self,contour):
        cont_area = cv.contourArea(contour) 
        (x,y), radius = cv.minEnclosingCircle(contour)
        encl_area = math.pi * radius * radius
        if (cont_area/encl_area)*100 > self.min_percent_filled and cont_area >= self.min_area:
            return True
        return False
    
    def getFrame(self):
        ret, frame = self.cap.read()
        height, width, n = frame.shape
        frame = frame[:, :width//2]
        cv.imshow('frame',frame)
        return frame
    
    def killCap(self):
        self.cap.release()

    def contour_find_ball(self,frame):

        ######### Blob detection #########

        if self.show_time:
            start_time = time.time()

        # Convert to HSV
        hsv_frame = cv.cvtColor(frame, cv.COLOR_BGR2HSV)

        # Apply adaptive mask to HSV frame if lock is on
        if self.lock_on and self.use_lock:
            height, width, n = frame.shape
            prev_detection_mask = np.zeros((height, width), dtype=np.uint8)
            cv.circle(prev_detection_mask, (int(self.x_est),int(self.y_est)), self.lock_radius * 2, 255, -1)
            hsv_frame = cv.bitwise_and(hsv_frame, hsv_frame, mask = prev_detection_mask)

        # If ballon is close increase value
        if self.radius > 40:
            self.purple_uv = 250
            self.green_uv = 250

        # Create masks
        g_upper = np.array([self.green_uh,self.green_us,self.green_uv])
        g_lower = np.array([self.green_lh,self.green_ls,self.green_lv])
        p_upper = np.array([self.purple_uh,self.purple_us,self.purple_uv])
        p_lower = np.array([self.purple_lh,self.purple_ls,self.purple_lv])

        # Extend bounds if lock is on
        if self.lock_on:
            g_upper += 5
            g_lower -= 5
            p_upper += 5
            p_lower -= 5
            
        p_mask = cv.inRange(hsv_frame, p_lower, p_upper)
        g_mask = cv.inRange(hsv_frame, g_lower, g_upper)

        # Filter out green or purple if turned off
        if not self.include_green:
            g_mask = np.zeros_like(g_mask)
        if not self.include_purple:
            p_mask = np.zeros_like(p_mask)

        # Combine green and purple masks
        combined_mask = g_mask + p_mask

        # Erode and dilate mask
        if self.lock_on:
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
            if cont_area > max_area and self.isValidArea(contours[i]):
                max_area = cont_area
                best_i = i

        if best_i > -1:
            (self.x,self.y), self.radius = cv.minEnclosingCircle(contours[best_i])
            self.no_detect_count = 0
        else:
            self.x = -1
            self.y = -1
            self.no_detect_count += 1

        # Print blob detection time
        if self.show_time and self.start_time is not None:
            end_time = time.time()
            elapsed_time1 = end_time - start_time
            print('\nBlob detection: ', elapsed_time1)

            start_time = time.time()

        ######## Optical flow ########

        if self.use_optical_flow:
            if self.x > -1:
                if self.old_gray is None:
                    self.old_gray = cv.cvtColor(frame,cv.COLOR_BGR2GRAY)

                height, width, n = frame.shape
                optical_flow_mask = np.zeros((height, width), dtype=np.uint8)
                cv.circle(optical_flow_mask, (int(self.x),int(self.y)), int(self.radius), 255, -1)
                self.p0 = cv.goodFeaturesToTrack(self.old_gray, mask = optical_flow_mask, **self.feature_params)

            if self.p0 is not None:
                frame_sparse = frame.copy()
                frame_gray = cv.cvtColor(frame_sparse, cv.COLOR_BGR2GRAY)
                p1, st, err = cv.calcOpticalFlowPyrLK(self.old_gray, frame_gray, self.p0, None, **self.lk_params)

                # Select good points
                if p1 is not None:
                    good_new = p1[st==1]
                    good_old = self.p0[st==1]

                # Calc velocity
                feat_count = 0
                self.v_x_s = 0
                self.v_y_s = 0
                for i, (new, old) in enumerate(zip(good_new, good_old)):
                    a, b = new.ravel()
                    c, d = old.ravel()
                    self.v_x_s += c - a
                    self.v_y_s += b - d
                    feat_count += 1

                if not feat_count == 0:
                    self.v_x_s /= feat_count
                    self.v_y_s /= feat_count

                    self.v_x_s *= self.v_scale
                    self.v_y_s *= self.v_scale

                    if abs(self.v_x_s) > 5:
                        self.v_x_s = 0

                    if abs(self.v_y_s) > 5:
                        self.v_y_s = 0

                # Now update the previous frame and previous points
                self.old_gray = frame_gray.copy()
                self.p0 = good_new.reshape(-1, 1, 2)

        # Print optical flow time
        if self.show_time and self.start_time is not None:
            end_time = time.time()
            elapsed_time2 = end_time - start_time
            print('Optical flow: ', elapsed_time2)

            start_time = time.time()

        ######## Kalman filter ########

        if self.kalman_initialized and self.use_kalman:
            # Estimate
            Q = np.matrix([[self.Q_pos,0,0,0],[0,self.Q_pos,0,0],[0,0,self.Q_v,0],[0,0,0,self.Q_v]])
            dt = 1
            F_t = np.matrix([[1,0,dt,0], [0,1,0,dt], [0,0,1,0], [0,0,0,1]])

            # Predict
            x_hat = F_t @ self.x_prev
            P_t = F_t @ self.P_prev @ F_t.getT() + Q

            # Measurement
            if self.use_optical_flow:
                v_x = self.v_x_s
                v_y = self.v_y_s
            elif self.prev_x > -1:
                v_x = self.x - self.prev_x
                v_y = self.y - self.prev_y

            # Update
            if self.x > -1:
                z_t = np.matrix([[self.x],[self.y],[v_x],[v_y]])
                H_t = np.matrix([[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1]])
                R = np.matrix([[self.R_pos,0,0,0],[0,self.R_pos,0,0],[0,0,self.R_v,0],[0,0,0,self.R_v]])
                I = np.matrix([[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1]])
                res = z_t - (H_t @ self.x_prev)
                S = H_t @  P_t @ H_t.getT() + R
                K = P_t @ H_t.getT() @ S.getI()
                x_hat = x_hat + K @ res
                P_t = (I - K @ H_t) @ P_t

            self.x_est = x_hat[0].item()
            self.y_est = x_hat[1].item()

            self.P_prev = P_t
            self.x_prev = x_hat

            if self.no_detect_count > 100 or (self.x > -1 and (abs(self.x_est - self.x) > 10 and abs(self.y_est - self.y) > 10)):
                kalman_initialized = False
                self.x_est = -1
                self.y_est = -1

        elif self.x > -1:
                self.x_prev = np.matrix([[self.x],[self.y],[0],[0]])
                self.P_prev = np.matrix([[10,0,0,0],[0,10,0,0],[0,0,10,0],[0,0,0,10]])
                self.kalman_initialized = True

        # Count for how many frames the detection is in a similar area to prev detection
        if self.x_est > - 1 and abs(self.x_est - self.prev_x_est) < 5 and abs(self.y_est - self.prev_y_est) < 5:
            self.sim_detect_count += 1
        else:
            self.sim_detect_count = 0
            self.lock_on = False

        # Enable lock if balloon is in similar place for 50 frames
        if self.sim_detect_count > 20:
            self.lock_on = True

        # Update previous values
        self.prev_x = self.x
        self.prev_y = self.y
        self.prev_x_est = self.x_est
        self.prev_y_est = self.y_est

        # Time for kalman filter and total time
        if self.show_time and self.start_time is not None:
            end_time = time.time()
            elapsed_time3 = end_time - start_time
            print('Kalman: ', elapsed_time3)
            total_time = elapsed_time1 + elapsed_time2 + elapsed_time3
            print('Total: ', total_time)
            print('fps: ', 1/total_time)

        # Streaming
        # cv.circle(frame, (int(x_est),int(y_est)), int(radius), (0, 0, 255), 3)

        ######## Return detection ########
        diameter = self.radius * 2

        detection_msg = Detection()
        detection_msg.class_id = 1
        detection_msg.obj_class = "Shape"
        detection_msg.bbox[0] = self.x_est
        detection_msg.bbox[1] = self.y_est
        detection_msg.bbox[2] = diameter
        detection_msg.bbox[3] = diameter
        detection_msg.depth = -1.0
        detection_msg.confidence = -1.0
        detection_msg.track_id = -1

        return detection_msg


