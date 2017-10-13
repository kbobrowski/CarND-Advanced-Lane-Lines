import numpy as np
import cv2
from glob import glob
import matplotlib.pyplot as plt
import os


class LaneFinder(object):
    def __init__(self, calibration_images):
        self.calibrate(calibration_images)
        self.left_fit = None
        self.right_fit = None


    def calibrate(self, images):
        objp = np.zeros((6*9,3), np.float32)
        objp[:,:2] = np.mgrid[0:9,0:6].T.reshape(-1,2)

        objpoints = []
        imgpoints = []
        for fname in images:
            img = cv2.imread(fname)
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

            ret, corners = cv2.findChessboardCorners(gray, (9,6), None)

            if ret == True:
                objpoints.append(objp)
                imgpoints.append(corners)

                if False:
                    img = cv2.drawChessboardCorners(img, (9,6), corners, ret)
                    cv2.imshow('img', img)
                    cv2.waitKey(500)

        ret, self.mtx, self.dist, self.rvecs, self.tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)
       

    @staticmethod
    def perspective_transform(img, **kwargs):
        testrun = kwargs.get('testrun', False)
        src_points = kwargs.get('src_points')
        dst_points = kwargs.get('dst_points')
        M = cv2.getPerspectiveTransform(src_points, dst_points)
        warped = cv2.warpPerspective(img, M, (img.shape[1], img.shape[0]))

        if testrun:
            ax1 = kwargs.get('ax1')
            ax2 = kwargs.get('ax2')
            ax1.imshow(img)
            src = np.append(src_points, src_points[0][np.newaxis,:] , axis=0)
            dst = np.append(dst_points, dst_points[0][np.newaxis,:] , axis=0)
            ax1.plot(src[:,0], src[:,1], color='red')
            ax2.imshow(np.uint8(warped))
            ax2.plot(dst[:,0], dst[:,1], color='red')

        return warped


    @staticmethod
    def binary_image(img, **kwargs):
        testrun = kwargs.get('testrun', False)
        s_thresh = kwargs.get('s_thresh', (170, 255))
        sx_thresh = kwargs.get('sx_thresh', (20, 100))

        hls = cv2.cvtColor(img, cv2.COLOR_RGB2HLS).astype(np.float)
        l_channel = hls[:,:,1]
        s_channel = hls[:,:,2]
        sobelx = cv2.Sobel(l_channel, cv2.CV_64F, 1, 0) 
        abs_sobelx = np.absolute(sobelx) 
        scaled_sobel = np.uint8(255*abs_sobelx/np.max(abs_sobelx))

        # x gradient threshold
        sxbinary = np.zeros_like(scaled_sobel)
        sxbinary[(scaled_sobel >= sx_thresh[0]) & (scaled_sobel <= sx_thresh[1])] = 1

        # color channel threshold
        s_binary = np.zeros_like(s_channel)
        s_binary[(s_channel >= s_thresh[0]) & (s_channel <= s_thresh[1])] = 1

        color_binary = np.dstack((np.zeros_like(sxbinary), sxbinary, s_binary)) * 255

        if testrun:
            ax = kwargs.get('ax')
            ax.imshow(np.uint8(color_binary))

        return color_binary


    @staticmethod
    def combine_binary(img, **kwargs):
        testrun = kwargs.get('testrun', False)
        combined_binary = np.zeros_like(img[:,:,0])
        combined_binary[(img[:,:,1] == 255) | (img[:,:,2] == 255)] = 1

        if testrun:
            ax = kwargs.get('ax')
            ax.imshow(np.uint8(combined_binary*255), cmap='gray')

        return combined_binary


    def find_lines_init(self, binary_warped, **kwargs):
        testrun = kwargs.get('testrun', False)

        histogram = np.sum(binary_warped[int(binary_warped.shape[0]/2):,:], axis=0)
        out_img = np.dstack((binary_warped, binary_warped, binary_warped))*255
        midpoint = np.int(histogram.shape[0]/2)
        leftx_base = np.argmax(histogram[:midpoint])
        rightx_base = np.argmax(histogram[midpoint:]) + midpoint

        nwindows = 9
        window_height = np.int(binary_warped.shape[0]/nwindows)
        nonzero = binary_warped.nonzero()
        nonzeroy = np.array(nonzero[0])
        nonzerox = np.array(nonzero[1])
        leftx_current = leftx_base
        rightx_current = rightx_base
        margin = 100
        minpix = 50
        left_lane_inds = []
        right_lane_inds = []

        for window in range(nwindows):
            win_y_low = binary_warped.shape[0] - (window+1)*window_height
            win_y_high = binary_warped.shape[0] - window*window_height
            win_xleft_low = leftx_current - margin
            win_xleft_high = leftx_current + margin
            win_xright_low = rightx_current - margin
            win_xright_high = rightx_current + margin
            cv2.rectangle(out_img,(win_xleft_low,win_y_low),(win_xleft_high,win_y_high),
                                   (0,255,0), 2) 
            cv2.rectangle(out_img,(win_xright_low,win_y_low),(win_xright_high,win_y_high),
                                   (0,255,0), 2) 
            good_left_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & 
                              (nonzerox >= win_xleft_low) &  (nonzerox < win_xleft_high)).nonzero()[0]
            good_right_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & 
                               (nonzerox >= win_xright_low) &  (nonzerox < win_xright_high)).nonzero()[0]
            left_lane_inds.append(good_left_inds)
            right_lane_inds.append(good_right_inds)
            if len(good_left_inds) > minpix:
                leftx_current = np.int(np.mean(nonzerox[good_left_inds]))
            if len(good_right_inds) > minpix:        
                rightx_current = np.int(np.mean(nonzerox[good_right_inds]))

        left_lane_inds = np.concatenate(left_lane_inds)
        right_lane_inds = np.concatenate(right_lane_inds)

        self.leftx = nonzerox[left_lane_inds]
        self.lefty = nonzeroy[left_lane_inds] 
        self.rightx = nonzerox[right_lane_inds]
        self.righty = nonzeroy[right_lane_inds] 

        self.left_fit = np.polyfit(self.lefty, self.leftx, 2)
        self.right_fit = np.polyfit(self.righty, self.rightx, 2)

        if False and testrun:
            ax = kwargs.get('ax')
            ploty = np.linspace(0, binary_warped.shape[0]-1, binary_warped.shape[0] )
            left_fitx = self.left_fit[0]*ploty**2 + self.left_fit[1]*ploty + self.left_fit[2]
            right_fitx = self.right_fit[0]*ploty**2 + self.right_fit[1]*ploty + self.right_fit[2]

            out_img[nonzeroy[left_lane_inds], nonzerox[left_lane_inds]] = [255, 0, 0]
            out_img[nonzeroy[right_lane_inds], nonzerox[right_lane_inds]] = [0, 0, 255]
            ax.imshow(np.uint8(out_img))
            ax.plot(left_fitx, ploty, color='yellow')
            ax.plot(right_fitx, ploty, color='yellow')
            ax.set_xlim(0, 1280)
            ax.set_ylim(720, 0)


    def find_lines(self, binary_warped, **kwargs):
        testrun = kwargs.get('testrun', False)
        if self.left_fit is None:
            self.find_lines_init(binary_warped, **kwargs)
            self.find_lines(binary_warped, **kwargs)
            return
        nonzero = binary_warped.nonzero()
        nonzeroy = np.array(nonzero[0])
        nonzerox = np.array(nonzero[1])
        margin = 100
        left_lane_inds = ((nonzerox > (self.left_fit[0]*(nonzeroy**2) + self.left_fit[1]*nonzeroy + self.left_fit[2] - margin)) & (nonzerox < (self.left_fit[0]*(nonzeroy**2) + self.left_fit[1]*nonzeroy + self.left_fit[2] + margin))) 

        right_lane_inds = ((nonzerox > (self.right_fit[0]*(nonzeroy**2) + self.right_fit[1]*nonzeroy + self.right_fit[2] - margin)) & (nonzerox < (self.right_fit[0]*(nonzeroy**2) + self.right_fit[1]*nonzeroy + self.right_fit[2] + margin)))  

        self.leftx = nonzerox[left_lane_inds]
        self.lefty = nonzeroy[left_lane_inds] 
        self.rightx = nonzerox[right_lane_inds]
        self.righty = nonzeroy[right_lane_inds]
        self.left_fit = np.polyfit(self.lefty, self.leftx, 2)
        self.right_fit = np.polyfit(self.righty, self.rightx, 2)

        if testrun:
            ax = kwargs.get('ax')
            ploty = np.linspace(0, binary_warped.shape[0]-1, binary_warped.shape[0] )
            self.ploty = ploty
            left_fitx = self.left_fit[0]*ploty**2 + self.left_fit[1]*ploty + self.left_fit[2]
            right_fitx = self.right_fit[0]*ploty**2 + self.right_fit[1]*ploty + self.right_fit[2]
            self.left_fitx = left_fitx
            self.right_fitx = right_fitx
            
            # Create an image to draw on and an image to show the selection window
            out_img = np.dstack((binary_warped, binary_warped, binary_warped))*255
            window_img = np.zeros_like(out_img)
            # Color in left and right line pixels
            out_img[nonzeroy[left_lane_inds], nonzerox[left_lane_inds]] = [255, 0, 0]
            out_img[nonzeroy[right_lane_inds], nonzerox[right_lane_inds]] = [0, 0, 255]

            # Generate a polygon to illustrate the search window area
            # And recast the x and y points into usable format for cv2.fillPoly()
            left_line_window1 = np.array([np.transpose(np.vstack([left_fitx-margin, ploty]))])
            left_line_window2 = np.array([np.flipud(np.transpose(np.vstack([left_fitx+margin, 
                                          ploty])))])
            left_line_pts = np.hstack((left_line_window1, left_line_window2))
            right_line_window1 = np.array([np.transpose(np.vstack([right_fitx-margin, ploty]))])
            right_line_window2 = np.array([np.flipud(np.transpose(np.vstack([right_fitx+margin, 
                                          ploty])))])
            right_line_pts = np.hstack((right_line_window1, right_line_window2))

            # Draw the lane onto the warped blank image
            cv2.fillPoly(window_img, np.int_([left_line_pts]), (0,255, 0))
            cv2.fillPoly(window_img, np.int_([right_line_pts]), (0,255, 0))
            result = cv2.addWeighted(out_img, 1, window_img, 0.3, 0)
            ax.imshow(np.uint8(result))
            ax.plot(left_fitx, ploty, color='yellow')
            ax.plot(right_fitx, ploty, color='yellow')
            ax.set_xlim(0, 1280)
            ax.set_ylim(720, 0)


    def measure_curvature(self):
        y_eval = np.max(self.ploty)
        # Define conversions in x and y from pixels space to meters
        ym_per_pix = 30/720 # meters per pixel in y dimension
        xm_per_pix = 3.7/700 # meters per pixel in x dimension

        # Fit new polynomials to x,y in world space
        left_fit_cr = np.polyfit(self.ploty*ym_per_pix, self.left_fitx*xm_per_pix, 2)
        right_fit_cr = np.polyfit(self.ploty*ym_per_pix, self.right_fitx*xm_per_pix, 2)
        # Calculate the new radii of curvature
        left_curverad = ((1 + (2*left_fit_cr[0]*y_eval*ym_per_pix + left_fit_cr[1])**2)**1.5) / np.absolute(2*left_fit_cr[0])
        right_curverad = ((1 + (2*right_fit_cr[0]*y_eval*ym_per_pix + right_fit_cr[1])**2)**1.5) / np.absolute(2*right_fit_cr[0])
        # Now our radius of curvature is in meters
        print(left_curverad, 'm', right_curverad, 'm')


    def run(self, img, **kwargs):
        testrun = kwargs.get('testrun', False)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        if testrun:
            f, ax = plt.subplots(2, 3)
            ax = ax.flatten()
            ax[0].imshow(img)
        else:
            ax = [None]*6
        
        undist = cv2.undistort(img, self.mtx, self.dist, None, self.mtx)
        warped = self.perspective_transform(undist, ax1=ax[1], ax2=ax[2], **kwargs)
        binary = self.binary_image(warped, ax=ax[3], **kwargs)
        combined_binary = self.combine_binary(binary, ax=ax[4], **kwargs)
        self.find_lines(combined_binary, ax=ax[5], **kwargs)
        #self.find_lines(combined_binary, ax=ax[5], **kwargs)
        self.measure_curvature()


        if testrun:
            plt.tight_layout()
            plt.show()

        #return np.uint8(combined_binary)


calibration_images = glob('camera_cal/calibration*.jpg')
rawimages = glob('test_images/*.jpg')

process = LaneFinder(calibration_images)

vidcap = cv2.VideoCapture('project_video.mp4')
success, image = vidcap.read()
print(success)
count = 0
success = True
import sys
sys.exit(0)
for i in range(1,7):
    img = cv2.imread(rawimages[i])
    process.run(img,
                testrun=True,
                src_points=np.float32([[120, 650],
                                       [570, 450],
                                       [710, 450],
                                       [1180, 650]]),
                dst_points=np.float32([[100, 700],
                                       [100, 10],
                                       [1200, 10],
                                       [1200, 700]]),
                s_thresh=(170, 255),
                sx_thresh=(20,100),
                y_eval=700)

#plt.imshow(img)
#plt.show()

import sys
sys.exit(0)
f, (ax1, ax2) = plt.subplots(1,2)
ax1.imshow(img)
ax2.imshow(np.uint8(pipeline(img)))
plt.show()

cv2.destroyAllWindows()
