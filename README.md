[![Advanced Lane Detection & Tracking - Self-Driving Car NanoDegree](https://s3.amazonaws.com/udacity-sdc/github/shield-carnd.svg)](http://www.udacity.com/drive)
## Dean Webb - Advanced Lane Detection & Tracking Pipeline
#### Self-Driving Car Engineer Nanodegee - Project 4
In this project, our goal is to write a software pipeline to identify the lane boundaries in an input video. An example summary of all techniques applied to a test image can be seen below for reference:

---

![alt text][image21]

---

![alt text][image11]

---
### <font color='green'>Project Goals</font>

The goals / steps of this project are the following:

* Compute the camera calibration matrix and distortion coefficients given a set of chessboard images.
* Apply a distortion correction to raw images.
* Use color transforms, gradients, etc., to create a thresholded binary image.
* Apply a perspective transform to rectify binary image ("birds-eye view").
* Detect lane pixels and fit to find the lane boundary.
* Determine the curvature of the lane and vehicle position with respect to center.
* Warp the detected lane boundaries back onto the original image.
* Output visual display of the lane boundaries and numerical estimation of lane curvature and vehicle position.

---

### Dependencies

The images for camera calibration are stored in the folder called `camera_cal`.  The images in `test_images` are for testing your pipeline on single frames.  If you want to extract more test images from the videos, you can simply use an image writing method like `cv2.imwrite()`, i.e., you can read the video in frame by frame as usual, and for frames you want to save for later you can write to an image file.  

To help the reviewer examine your work, please save examples of the output from each stage of your pipeline in the folder called `ouput_images`, and include a description in your writeup for the project of what each image shows.    The video called `project_video.mp4` is the video your pipeline should work well on.  

The `challenge_video.mp4` video is an extra (and optional) challenge for you if you want to test your pipeline under somewhat trickier conditions.  The `harder_challenge.mp4` video is another optional challenge and is brutal!

---

[//]: # (Image References)

[image1]: data/camera_cal/calibration1.jpg "Calibration Input"
[image8]: data/output_images/corners_found11.jpg "Undistorted"
[image2]: data/test_images/test1.jpg "Test Image Input"
[image4]: data/examples/out/apply_perspective_transform.jpg "Perspective Transform Function Snippet"
[image5]: data/output_images/green_lines5.jpg "Fit Visual (With Green Lines)"
[image6]: data/examples/out/example_processed_output.jpg "Radius Curvature Output"
[image7]: data/processed_project_video.mp4 "Processed Project Video"
[image9]: data/output_images/undistorted0.jpg "Test Image Input"
[image10]: data/examples/out/curvature_snippet.jpg "Radius of Curvature Code Snippet"
[image12]: data/output_images/thresholded1.jpg "Thresholded - Masked"
[image11]: data/examples/out/overview.jpg "Overview"
[image13]: data/examples/out/apply_thresholds.jpg "Apply Thresholds"
[image14]: data/output_images/thresholded4.jpg "Thresholded"
[image15]: data/output_images/color_stacked4.jpg "Color Stacked"
[image16]: data/test_images/test5.jpg "Test Image 5"
[image17]: data/output_images/thresholded2.jpg "Test Img 3 Thresholded"
[image18]: data/output_images/warped2.jpg "Warped 2"
[image19]: data/examples/out/inverted_warped_transform.jpg "Inverted Mask Warped 2"
[image20]: data/examples/out/histogram_analysis.jpg "Histogram Analysis Function"
[image21]: data/examples/out/overview_processed.jpg "Overview Lane-Tracking Processed"



# <font color='red'> Rubric Points</font>
### Here I will consider the [rubric](https://review.udacity.com/#!/rubrics/571/view) points individually and describe how I addressed each point in my implementation.  

---
### Writeup / README

#### 1. <font color='green'>Provide a Writeup / README</font> that includes all the rubric points and how you addressed each one.  You can submit your writeup as markdown or pdf.

**Done!** - *See below.*

### Camera Calibration

#### 1. Briefly state how you computed the camera matrix and distortion coefficients.  <font color='green'>Provide an example of a distortion corrected calibration image.</font>

* The code for this step is contained in the code cells of the attached IPython notebook. The output is also located in "data/output_images/corners_found11.jpg".  

* I start by preparing "object points", which will be the (x, y, z) coordinates of the chessboard corners in the world. Here I am assuming the chessboard is fixed on the (x, y) plane at z=0, such that the object points are the same for each calibration image.

* Thus, `objp` is just a replicated array of coordinates, and `objpoints` will be appended with a copy of it every time I successfully detect all chessboard corners in a test image.  `imgpoints` will be appended with the (x, y) pixel position of each of the corners in the image plane with each successful chessboard detection.

* I then used the output `objpoints` and `imgpoints` to compute the camera calibration and distortion coefficients using the `cv2.calibrateCamera()` function. An example of this process can be seen below:

---

|   Original    |  Undistorted  |
|:-------------:|:-------------:|
|  ![][image1]  |  ![][image8]  |

For added visual effect above, I added the color markings that were computed using `cv2.findChessboardCorners()` - (*See e.g., code cells* **3-4** of `advanced-lane-lines-setup.ipynb`).

---

### Lane-Finding Pipeline (single images)

#### 1. <font color='green'>Provide an example of a distortion-corrected test image.</font>
I also applied this distortion correction to the test image using the `cv2.undistort()` function and obtained this result:

---

|   Test Img      |  Undistorted Img |
|:---------------:|:---------------: |
|   ![][image2]   |   ![][image9]    |

---

#### 2. Describe how (and identify where in your code) you used color transforms, gradients or other methods to create a thresholded binary image.  <font color='green'>Provide an example of a binary image result.</font>
I used a combination of color and gradient thresholds to generate the binary image, as shown in the code snippet below for convenience. This snippet for applying the thresholds can also be seen in **code cell 12** of `advanced-lane-lines-setup.ipynb`).

---

![alt text][image13]

---

It took quite a long time to figure out the optimal configuration of thresholds to apply. In many ways the values are stil lnot perfect! However, the smoothing factor applied during video generation helps keep track of the lines once they are found. Conveniently, below are some examples of my generated output for my custom thresholding technique. I found the v_channel, s_channel, and l_channel color binary channels carried a lot of information with them, which helped in detection of the lines in the output throsholded image. Please note: The color stacked binary image is not included in the output, it is added here for illustration purposes.

|  Undistorted Img  |   Color Binaries |  Thresholded Img  |
|:-----------------:|:---------------: |:-----------------:|
|   ![][image16]    |   ![][image15]   |   ![][image14]    |

---


#### 3. Describe how (and identify where in your code) you performed a perspective transform and provide an example of a transformed image.

The code for my perspective transform includes a function called `apply_perspective_transform()`, which appears in the code snippet (which has been included below for the reviewer's convenience. This snippet for applying the perspective transform can also be seen in **code cell 13** of `advanced-lane-lines-setup.ipynb`).   The `apply_perspective_transform()` function takes as inputs an image (`img`). I also included (as an optional parameter) the M_pickle file, which contains the Transformation Matrix (`M`), as well as the Inverse Transformation Matrix (`Minv`), both of which are calculated by `warpPerspective()` from the OpenCV library. For our purposes, we will use this function with the flag `cv2.INTER_LINEAR`, which is described by the source as a [a bilinear interpolation.](http://docs.opencv.org/2.4/modules/imgproc/doc/geometric_transformations.html).

---

![alt text][image4]

---

By including the M_pickle file, the function can optionally bypass the recalculation of the source (`src`) and destination (`dst`) points if the file exists. As hinted by the name, we use this function to warp the image in such a way that it appearsto change its orientation. For example, we will use the image warping to map a bird's eye view of the lane lines so that it would be easier to detect the parallel nature of the lanes. But forst, we must capture the perspective in it's image space. I used a trapezoidal shape with the intention on transforming the points into a rectangular one. Instead of fully hardcoding the source and destination points, the function attempts to partially generate these source (`src`) and destination (`dst`) points based on the image size variables, and the desired trapezoidal destination region. As shown above, these weighted variables shape the trapezoid by plotting its:
* Bottom Trapezoidal Width (`bottom_width`)
* Middle Trapezoidal Width (`mid_width`)
* Percent of the Trapezoidal Height (`height_pct`)
* Percent to Trim from top to bottom - to remove the car's hood (`bottom_trim`)

By tuning these parameters, I verified that my perspective transform was working as expected by drawing the `src` and `dst` points onto a test image and its warped counterpart to verify that the lines appear parallel in the warped image. Below are examples of my generated output for this perspective transform technique.


|  Undistorted Img  | Persp. Trans Img | Inverted Masked   |
|:-----------------:|:---------------: |:-----------------:|
|   ![][image17]    |   ![][image18]   |   ![][image19]    |


#### 4. Describe how (and identify where in your code) you identified lane-line pixels and fit their positions with a polynomial?

In order to identify the laneline pixels. I used a convolutional technique, which can be found in the `tracker()` class in the `advanced-lane-lines-setup.ipynb` file. The tracker class advantageously has a function called `find_window_centroids` which does a lot of the heavy lifting involved with detecting windows. I found this technique to work much better than my original plan to use a Histogram of pixel in the vertical axis.Once the windows were detected by the tracker class. I did various array manipulations and concatenations to select the `left-lane` points and the `right-lane` points. Next, I further applied some matrix manipulations and region masking in order to fit the detected lane lines within a 2nd order polynomial. I note that the "lines" pixels of the second order polynomial are determined by plotting the windows gathered by the windows centroids from the `tracker` class. More specifically, I used the center of the windows and plotted them in green in order to find the line. Because the windows were detected as a convolution, the was much less noise than the histogram technique. For reference, below is a visualization of the resulting processed image, with the lines detected:

---

![alt text][image5]

---

The code for the above shown image can be found in the `pipeline()` function, which appears in a code cell with the header "Lane Tracking Pipeline." This function accepts a list of images and applies all computer vision techniques on the image to produce an overview (the same overview plotted at the beginning of this report.) In addition to the `pipeline()` function, I similarly created a `process_image()` function to be applied on a single image frame. The key difference between the two methods is that the `process_image()` function bypasses all of the visualization plots, debugger logging and image saving that exists in the `pipeline()` function.

#### 5. Describe how (and identify where in your code) you calculated the radius of curvature of the lane and the position of the vehicle with respect to center.

Following this handy [radius of curvature](http://www.intmath.com/applications-differentiation/8-radius-curvature.php) tutorial, I was able to apply the technique on the resulting image. Once these values where calculated I then plotted the values directly onto the image for verification that the lanes are in fact curved. As shown in the code snippet below for convenience, this snippet for calculating the radius of curvature is as follows:

---

![alt text][image10]

---

Note: This snippet is fully shown in various **code cells** of `advanced-lane-lines-setup.ipynb`. Particularly, the radius of curvature is calculated in the `pipeline()` and `process_image()` functions.

#### 6. <font color='green'>Provide an example image of your result</font> plotted back down onto the road such that the lane area is identified clearly.

As noted above, I implemented this step in the `pipeline()` and `process_image()` functions. Here is an example of my result on a test image:

---

![alt text][image6]

---

### Pipeline (video)

#### 1. <font color='green'>Provide a link to your final video output.</font>  Your pipeline should perform reasonably well on the entire project video (wobbly lines are ok but no catastrophic failures that would cause the car to drive off the road!).

Here's a [link to my video result](https://youtu.be/wmLn4muEI7k)

---

### Discussions / Learnings

#### 1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

Here I'll talk about the approach I took, what techniques I used, what worked and why, where the pipeline might fail and how I might improve it if I were going to pursue this project further.

The biggest problems I faced with implementation had to do with the `mpimg().imread` versus `cv2.imread()` functions for reading in images. Not only do these libraries differ in their orientation of the matrix (`BGR`) for Open CV functions, versus (`RGB`) for `impimg.imread()` functions. This took quite some time getting used to and I still have occasional hiccups with that implementation.

Another issue was that originally I used a Histogram to try and detect the polynomial fitted lines, but I found the method to not be very accurate. As such, in order to order to identify the lane pixels, I implemented the convolutional technique described above. However, I still found the Histogram function to be quite useful for debugging and tuning my parameters for region masking. As an example of how I used this, I include below an analysis function I utilized to tweak my parameters:

---

![alt text][image20]

---

Since I implemented. By far the best strategy I used was the `pipeline()` function, as it goes through and applies all of the required transformations and then saves the images into a directory. This strategy allowed my to slowly develop and to "fail fast." That is, I was able to recognize bugs in the code or mistakes in my logic by simply looking at the output image each step. I initially tried to implement the vehicle detection tracking project without this intermediary process of plotting and saving example images. I believe this will help me further limit the false positives that have been plaguing my results (and delaying my submission).

I am satisfied with my lane line detection accuracy but I definitely believe I could spend a great deal more time on the project perfecting that. I would like to investigate some techniques that might allow me to plot lanes for **very sharp turns** I quickly realized after testing with the `harder_challenge.mp4` video that there are many improvements to be made to the algorithm I used. I noticed many scenarios that easily broke my lane detection algorithms, such as the road changing color. To fix this, I would process a subclip of the video, and rocess and print each frame to see what the algorithm was doing. In one such instance,  noticed the algorithm picking up the right lane car's wheel whenever the dashed-right lane line was missing. This allowed me to appropriately remove the false positives and complete detection on pretty much all of the frames. I believe a further investigation is in order to look into optimizing may lane-detection algorithm. I am pretty motivated to complete this **sooner rather than later**, since I plan to participate in the [Didi and Udacity Self Driving Car Challenge](https://www.udacity.com/didi-challenge)!

One more optimization I think would be great is if I could design the code to obviously run faster. I was more concerned with accuracy than speed for this project. Now that things are fairly accurate, I would investigate how to get it to process faster than the single-image-per-second threshold.

Still further, I think this detection would be a prime candidate to apply deep learning techniques. That is to say, to train a convolutional neural network to take in the input as an image and predict a list of window centroids that are centered on the left and right lanes. In fact, I think that if I implement such an algorithm I would be able to satisfy all 3 of my optimization ideas.
