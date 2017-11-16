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
        self.left_hist = []
        self.right_hist = []


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
       

    def perspective_transform(self, img, **kwargs):
        testrun = kwargs.get('testrun', False)
        src_points = kwargs.get('src_points')
        dst_points = kwargs.get('dst_points')
        M = cv2.getPerspectiveTransform(src_points, dst_points)
        self.Minv = cv2.getPerspectiveTransform(dst_points, src_points)
        self.warped = cv2.warpPerspective(img, M, (img.shape[1], img.shape[0]))

        if testrun:
            ax1 = kwargs.get('ax1')
            ax2 = kwargs.get('ax2')
            ax1.imshow(img)
            src = np.append(src_points, src_points[0][np.newaxis,:] , axis=0)
            dst = np.append(dst_points, dst_points[0][np.newaxis,:] , axis=0)
            ax1.plot(src[:,0], src[:,1], color='red')
            ax2.imshow(np.uint8(self.warped))
            ax2.plot(dst[:,0], dst[:,1], color='red')

        return self.warped


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
        ploty = np.linspace(0, binary_warped.shape[0]-1, binary_warped.shape[0] )
        left_fitx = self.left_fit[0]*ploty**2 + self.left_fit[1]*ploty + self.left_fit[2]
        right_fitx = self.right_fit[0]*ploty**2 + self.right_fit[1]*ploty + self.right_fit[2]
        self.ploty = ploty
        self.left_fitx = left_fitx
        self.righ_fitx = right_fitx

        if False and testrun:
            ax = kwargs.get('ax')

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

        self.left_hist.append(self.left_fit)
        self.right_hist.append(self.right_fit)

        if len(self.left_hist) > 20: self.left_hist.pop(0)
        if len(self.right_hist) > 20: self.right_hist.pop(0)

        left_hist = np.vstack(self.left_hist)
        right_hist = np.vstack(self.right_hist)
        left_std = np.std(left_hist, axis=0)
        right_std = np.std(right_hist, axis=0)
        left_fit_av = left_hist.mean(axis=0)
        right_fit_av = right_hist.mean(axis=0)
       
        left_stability = left_std[:2].mean()
        right_stability = right_std[:2].mean()
        if left_stability+right_stability == 0:
            fit_mean = (left_fit_av[:2] + right_fit_av[:2])/2
        else:
            fit_mean = (left_stability*left_fit_av[:2] + right_stability*right_fit_av[:2])/(left_stability + right_stability)
        left_fit_av[:2] = (left_fit_av[:2] + fit_mean)/2
        right_fit_av[:2] = (right_fit_av[:2] + fit_mean)/2

        ploty = np.linspace(0, binary_warped.shape[0]-1, binary_warped.shape[0] )
        left_fitx = left_fit_av[0]*ploty**2 + left_fit_av[1]*ploty + left_fit_av[2]
        right_fitx = right_fit_av[0]*ploty**2 + right_fit_av[1]*ploty + right_fit_av[2]
        self.left_fit = left_fit_av
        self.right_fit = right_fit_av
        self.left_fitx = left_fitx
        self.right_fitx = right_fitx
        self.ploty = ploty

        if testrun:
            ax = kwargs.get('ax')
            #left_fitx = self.left_fit[0]*ploty**2 + self.left_fit[1]*ploty + self.left_fit[2]
            #right_fitx = self.right_fit[0]*ploty**2 + self.right_fit[1]*ploty + self.right_fit[2]
            
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
        cy = ym_per_pix
        cx = xm_per_pix

        correction = xm_per_pix*np.array([1/ym_per_pix**2,
                                          1/ym_per_pix,
                                          1])
        left_fit_cr = correction * self.left_fit
        right_fit_cr = correction * self.right_fit
        

        # Calculate the new radii of curvature
        left_curverad = ((1 + (2*left_fit_cr[0]*y_eval*ym_per_pix + left_fit_cr[1])**2)**1.5) / np.absolute(2*left_fit_cr[0])
        right_curverad = ((1 + (2*right_fit_cr[0]*y_eval*ym_per_pix + right_fit_cr[1])**2)**1.5) / np.absolute(2*right_fit_cr[0])
        return (left_curverad + right_curverad)/2
        # Now our radius of curvature is in meters
        #print(left_curverad, 'm', right_curverad, 'm'


    def final_plot(self, undist):
        # Create an image to draw the lines on
        warp_zero = np.zeros_like(self.warped)
        color_warp = np.dstack((warp_zero, warp_zero, warp_zero))

        # Recast the x and y points into usable format for cv2.fillPoly()
        pts_left = np.array([np.transpose(np.vstack([self.left_fitx, self.ploty]))])
        pts_right = np.array([np.flipud(np.transpose(np.vstack([self.right_fitx, self.ploty])))])
        pts = np.int32(np.hstack((pts_left, pts_right)))
        # Draw the lane onto the warped blank image
        color_warp = np.zeros((self.warped.shape[0],self.warped.shape[1],3)).astype(np.uint8)
        cv2.fillPoly(color_warp, [pts], (0,255, 0))

        # Warp the blank back to original image space using inverse perspective matrix (Minv)
        newwarp = cv2.warpPerspective(color_warp, self.Minv, (self.warped.shape[1], self.warped.shape[0])) 
        # Combine the result with the original image
        result = cv2.addWeighted(undist, 1, newwarp, 0.3, 0)
        return result

    
    def test_undistort(self, img):
        return cv2.undistort(img, self.mtx, self.dist, None, self.mtx)



    def run(self, img, **kwargs):
        testrun = kwargs.get('testrun', False)
        finalimage = kwargs.get('finalimage', False)
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
        curverad = self.measure_curvature()


        if testrun:
            framenum = kwargs.get('framenum')
            plt.tight_layout()
            plt.suptitle('frame {}'.format(framenum))
            f.canvas.draw()
            data = np.fromstring(f.canvas.tostring_rgb(), dtype=np.uint8, sep='')
            data = data.reshape(f.canvas.get_width_height()[::-1] + (3,)) 
            return data

        if finalimage:
            framenum = kwargs.get('framenum')
            frame = self.final_plot(undist)
            center = (self.left_fitx[-1] + self.right_fitx[-1])/2
            center_frame = frame.shape[1]/2
            xm_per_pix = 3.7/700 # meters per pixel in x dimension
            position = (center_frame-center)*xm_per_pix

            curverad = '{:.2f}'.format(curverad) if curverad < 3000 else '> 3000'
            
            cv2.rectangle(frame, (30,10), (450,160), (255,255,255), -1)
            cv2.putText(frame, "frame: {}".format(framenum), (50,50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,0), 2)
            cv2.putText(frame, "curvature: {} m".format(curverad), (50,90), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,0), 2)
            cv2.putText(frame, "position: {:.2f} m".format(position), (50,130), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,0), 2)



            #plt.tight_layout()
            #plt.title('frame {}, curvature {}'.format(framenum, curverad, )) 
            #f.canvas.draw()
            #data = np.fromstring(f.canvas.tostring_rgb(), dtype=np.uint8, sep='')
            #data = data.reshape(f.canvas.get_width_height()[::-1] + (3,)) 
            return frame

        #return np.uint8(combined_binary)


calibration_images = glob('camera_cal/calibration*.jpg')
rawimages = glob('test_images/*.jpg')

process = LaneFinder(calibration_images)

from tunepy import tunepy, tunable

@tunepy
def get_frame(frame_number, s_thresh_low, s_thresh_high, sx_thresh_low, sx_thresh_high):
    img = np.load('frames/{}.npy'.format(frame_number))
    frame = process.run(img,
                        testrun=True,
                        src_points=np.float32([[120, 650],
                                               [570, 450],
                                               [710, 450],
                                               [1180, 650]]),
                        dst_points=np.float32([[100, 700],
                                               [100, 10],
                                               [1200, 10],
                                               [1200, 700]]),
                        s_thresh=(s_thresh_low, s_thresh_high),
                        sx_thresh=(sx_thresh_low, sx_thresh_high),
                        framenum=frame_number)

frame_number = tunable(int, [0, 1256])
s_thresh_low = tunable(int, [0, 255])
s_thresh_high = tunable(int, [0, 255])
sx_thresh_low = tunable(int, [0, 255])
sx_thresh_high = tunable(int, [0, 255])
#get_frame(frame_number, s_thresh_low, s_thresh_high,
#          sx_thresh_low, sx_thresh_high)

def process_video(fname, fout, finalimage=False, testrun=False):
    vidcap = cv2.VideoCapture(fname)
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    if testrun: shape = (640,480)
    elif finalimage: shape = (1280,720)
    out = cv2.VideoWriter(fout,fourcc, 20.0, shape)
    count = 0
    while True:
        success, img = vidcap.read()
        if not success: break
        frame = process.run(img,
                            finalimage=finalimage,
                            testrun=testrun,
                            src_points=np.float32([[120, 650],
                                                   [570, 450],
                                                   [710, 450],
                                                   [1180, 650]]),
                            dst_points=np.float32([[100, 700],
                                                   [100, 10],
                                                   [1200, 10],
                                                   [1200, 700]]),
                            s_thresh=(107, 138),
                            sx_thresh=(17,80),
                            framenum=count)
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        out.write(frame)
        count += 1
        print(count)
        #if count == 15: break
        plt.close()
        cv2.imshow('frame', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'): break
    vidcap.release()
    out.release()

#process_video('project_video.mp4', 'project_video_test_av4.mp4', testrun=True)
process_video('project_video.mp4', 'project_video_final2.mp4', finalimage=True)

#img = get_frame('project_video.mp4', 50)
#img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

import sys
sys.exit(0)
f, (ax1, ax2) = plt.subplots(1,2)
ax1.imshow(img)
ax2.imshow(np.uint8(pipeline(img)))
plt.show()

cv2.destroyAllWindows()
