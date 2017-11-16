## Writeup Template

### You can use this file as a template for your writeup if you want to submit it as a markdown file, but feel free to use some other method and submit a pdf if you prefer.

---

**Advanced Lane Finding Project**

The goals / steps of this project are the following:

* Compute the camera calibration matrix and distortion coefficients given a set of chessboard images.
* Apply a distortion correction to raw images.
* Use color transforms, gradients, etc., to create a thresholded binary image.
* Apply a perspective transform to rectify binary image ("birds-eye view").
* Detect lane pixels and fit to find the lane boundary.
* Determine the curvature of the lane and vehicle position with respect to center.
* Warp the detected lane boundaries back onto the original image.
* Output visual display of the lane boundaries and numerical estimation of lane curvature and vehicle position.

[//]: # (Image References)

[image1]: ./test_undistort.jpg "Test image"
[image1.1]: ./camera_cal/calibration1.jpg "Undistorted"
[image2]: ./test_images/test1.jpg "Road Transformed"
[image2.1]: ./test_images/test1_undistort.jpg "Road Transformed"
[image3]: ./process_images.png "Process image"
[image4]: ./final_example.png "Output"
[video1]: ./project_video.mp4 "Video"

## [Rubric](https://review.udacity.com/#!/rubrics/571/view) Points

### Here I will consider the rubric points individually and describe how I addressed each point in my implementation.  

---

### Writeup / README

#### 1. Provide a Writeup / README that includes all the rubric points and how you addressed each one.  You can submit your writeup as markdown or pdf.  [Here](https://github.com/udacity/CarND-Advanced-Lane-Lines/blob/master/writeup_template.md) is a template writeup for this project you can use as a guide and a starting point.  

You're reading it!

### Camera Calibration

#### 1. Briefly state how you computed the camera matrix and distortion coefficients. Provide an example of a distortion corrected calibration image.

The code for this step is contained in the "calibrate" method of LaneFinder class (line 17 in pipeline.py).  

I start by preparing "object points", which will be the (x, y, z) coordinates of the chessboard corners in the world. Here I am assuming the chessboard is fixed on the (x, y) plane at z=0, such that the object points are the same for each calibration image.  Thus, `objp` is just a replicated array of coordinates, and `objpoints` will be appended with a copy of it every time I successfully detect all chessboard corners in a test image.  `imgpoints` will be appended with the (x, y) pixel position of each of the corners in the image plane with each successful chessboard detection.  

I then used the output `objpoints` and `imgpoints` to compute the camera calibration and distortion coefficients using the `cv2.calibrateCamera()` function.  I applied this distortion correction to the test image using the `cv2.undistort()` function and obtained this result: 

![alt text][image1.1]
![alt text][image1]

### Pipeline (single images)

#### 1. Provide an example of a distortion-corrected image.

Followig example shows original and distortion-corrected image using undistort method:
![alt text][image2]
![alt text][image2.1]

### Pipeline overview

Following picture shows perspective transform, binary image and line detection (and is referred by the points below). This is a frame from a [full video available here.](./project_video_test_av4.mp4)

![alt text][image3]

#### 2. Describe how (and identify where in your code) you performed a perspective transform and provide an example of a transformed image.

The code for my perspective transform includes a method called `perspective_transform`, which appears in lines 41 through 59 in the file `pipeline.py`. The `perspective_transform` function takes as inputs an image (`img`), as well as source (`src_points`) and destination (`dst_points`) points.  I chose the hardcode the source and destination points in the following manner:

| Source        | Destination   | 
|:-------------:|:-------------:| 
| 120, 650      | 100, 700      | 
| 570, 450      | 100, 10       |
| 710, 450      | 1200, 10      |
| 1180, 650     | 1200, 700     |

I verified that my perspective transform was working as expected by drawing the `src` and `dst` points onto a test image and its warped counterpart to verify that the lines appear parallel in the warped image.

Example image: see image above.

#### 3. Describe how (and identify where in your code) you used color transforms, gradients or other methods to create a thresholded binary image.  Provide an example of a binary image result.

I used a combination of color and gradient thresholds to generate a binary image (thresholding steps in the method `binary_image` at line 63 in `pipeline.py`).

Example image: see image above. 

#### 4. Describe how (and identify where in your code) you identified lane-line pixels and fit their positions with a polynomial?

Based on a binary image, lane lines are extracted using `find_lines_init` method (line 105 in `pipeline.py`). 9 windows sliding along horizontal axis are used to determine line location, and lines are fitted using `np.polyfit`. Afterwards, in the subsequent frames lane lines are updated using `find_lines` method (line 177). Moreover, lines are stabilized using moving average. Final curvatures of both lines are the same (and equal to weighted average of left and right curvature, with weights proportional to each line stability over time, measured using standard deviation).

Example image: see image above.

#### 5. Describe how (and identify where in your code) you calculated the radius of curvature of the lane and the position of the vehicle with respect to center.

Curvature is computed using `measure_curvature` method in `pipeline.py` (line 263), and is equal to curvature of lines fit (quadratic function)  measured at the point closest to the car.

Position is determined by shift of the left line with respect to the center of the image (line 347 in `pipeline.py`).

#### 6. Provide an example image of your result plotted back down onto the road such that the lane area is identified clearly.

I implemented this step in lines 300 through 302 in my code in `pipeline.py` in the method `final_plot`.  Here is an example of my result on a test image:

![alt text][image4]

---

### Pipeline (video)

#### 1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (wobbly lines are ok but no catastrophic failures that would cause the car to drive off the road!).

Here's a [link to my video result](./project_video_final2.mp4)

---

### Discussion

#### 1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

The main issue was stability of the lines. I solved it by using averaging coefficients determining line shape over time. Another problem was that sometimes top of the line was not isolated properly in the binary image, which resulted in a different curvature of both lines. It was solved by forcing curvatures of both lines to be the same by averaging line coefficients. Weights used for averaging are proportional to current line stability (measured using standard deviation).

Pipeline will probably fail if only one line is visible, algorithm could be changed to determine where is the interior of the line in this case. Another improvement could be allowing lane to be have non-central location, which would make the pipeline more robust in case of lane change.
