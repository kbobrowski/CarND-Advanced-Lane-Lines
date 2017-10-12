import numpy as np
import cv2
from glob import glob
import matplotlib.pyplot as plt
import os

objp = np.zeros((6*9,3), np.float32)
objp[:,:2] = np.mgrid[0:9,0:6].T.reshape(-1,2)

objpoints = []
imgpoints = []

images = glob('camera_cal/calibration*.jpg')

# camera calibration
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

ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)

rawimages = glob('test_images/*.jpg')

# removing distorsion
for fname in rawimages:
    if False:
        img = cv2.imread(fname)
        dst = cv2.undistort(img, mtx, dist, None, mtx)
        #cv2.imshow('img', dst)
        #cv2.waitKey(500)
        plt.imsave("test_images_undist/{}".format(os.path.split(fname)[-1]), cv2.cvtColor(dst, cv2.COLOR_BGR2RGB))
        #plt.imsave("test.jpg".format(fname), dst)

# binary image
def binary_image(img):
    hls = cv2.cvtColor(img, cv2.COLOR_RGB2HLS).astype(np.float)
    l_channel = hls[:,:,1]
    s_channel = hls[:,:,2]
    sobelx = cv2.Sobel(l_channel, cv2.CV_64F, 1, 0) 
    abs_sobelx = np.absolute(sobelx) 
    scaled_sobel = np.uint8(255*abs_sobelx/np.max(abs_sobelx))

    # x gradient threshold
    sxbinary = np.zeros_like(scaled_sobel)
    sxbinary[(scaled_sobel >= 20) & (scaled_sobel <= 100)] = 1

    # color channel threshold
    s_binary = np.zeros_like(s_channel)
    s_binary[(s_channel >= 170) & (s_channel <= 255)] = 1

    color_binary = np.dstack((np.zeros_like(sxbinary), sxbinary, s_binary)) * 255

    return color_binary

img = cv2.imread(rawimages[0])
binary = binary_image(img)
print(binary[:,:,0])

f, (ax1, ax2) = plt.subplots(1,2)
ax1.imshow(img)
nim = np.zeros_like(img)+255
print(np.uint8(binary))
ax2.imshow(np.uint8(binary))
plt.show()

cv2.destroyAllWindows()
