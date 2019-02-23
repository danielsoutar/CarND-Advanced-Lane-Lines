## Writeup
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

[image1]: ./output_images/calibration.jpg "Calibration"
[image2]: ./output_images/undistorted1.jpg "Undistorted"
[image3]: ./output_images/thresholded1.jpg "Thresholded"
[image4]: ./output_images/warped1.jpg "Warped"
[image5]: ./output_images/display6.jpg "Boxes and pixels"
[image6]: ./output_images/mapped_region5.jpg "Mapped region"
[video1]: ./output_video/output_video.mp4 "Video"

## [Rubric](https://review.udacity.com/#!/rubrics/571/view) Points

### Here I will consider the rubric points individually and describe how I addressed each point in my implementation. Note that any images or the output video are available within the notebook, the video being in the 'output_video' folder.

---

### Writeup / README

#### 1. Provide a Writeup / README that includes all the rubric points and how you addressed each one.  You can submit your writeup as markdown or pdf. 

You're reading it!

### Camera Calibration

#### 1. Briefly state how you computed the camera matrix and distortion coefficients. Provide an example of a distortion corrected calibration image.

The code for this step is contained in the first few code cells of the IPython notebook located in "./submission/project.ipynb".

I start by preparing "object points", which will be the (x, y, z) coordinates of the chessboard corners in the world. Here I am assuming the chessboard is fixed on the (x, y) plane at z=0, such that the object points are the same for each calibration image.  Thus, `objp` is just a replicated array of coordinates, and `objpoints` will be appended with a copy of it every time I successfully detect all chessboard corners in a test image.  `imgpoints` will be appended with the (x, y) pixel position of each of the corners in the image plane with each successful chessboard detection.  

I then used the output `objpoints` and `imgpoints` to compute the camera calibration and distortion coefficients using the `cv2.calibrateCamera()` function.  I applied this distortion correction to the calibration test image using the `cv2.undistort()` function and obtained this result: 

![alt text][image1]

### Pipeline (single images)

#### 1. Provide an example of a distortion-corrected image.

To demonstrate this step, I will describe how I apply the distortion correction to one of the test images like this one:
![alt text][image2]

Roughly, I read in an image exactly as I did above with the chessboard images, and using the same calibration coefficients as before, I apply them and obtain the undistorted image. Usefully, since I already performed my calibration, I only need do this once, and can use the same camera matrix M on subsequent images, a simple and useful optimisation to reduce computational overhead.

#### 2. Describe how (and identify where in your code) you used color transforms, gradients or other methods to create a thresholded binary image.  Provide an example of a binary image result.

In the end I simply used colour thresholds to generate a binary image, with the thresholding code residing in 'Threshold.py'. In my case it seemed that the other channels gave too much noise, particularly those pertaining to gradients. I ended up using the B-channel from the LAB colour space and the L-channel from the HLS colour space, which were good at distinguishing between yellow and white lines respectively, irrespective of shadow. This appears to be similar to what others were reporting as well - there are certain points regarding gradient I will reflect on later. Here's an example of my output for this step:

![alt text][image3]

#### 3. Describe how (and identify where in your code) you performed a perspective transform and provide an example of a transformed image.

The code for my perspective transform includes a function called `transform()`, which appears in the middle of the IPython notebook.  The `transform()` function takes as inputs an image (`img`), with the transformation matrix M having been defined once by the src and dst points, which were determined above.  I chose to hardcode the source and destination points in the following manner:

```python
src = np.float32(
    [[560, 465],
    [730, 465],
    [1120, 715],
    [160, 715]])
dst = np.float32(
    [[image.shape[1] / 4, 0],
    [(image.shape[1] / 4) * 3, 0],
    [(image.shape[1] / 4) * 3, image.shape[0]],
    [image.shape[1] / 4, image.shape[0]]])
```

This resulted in the following source and destination points:

| Source        | Destination   | 
|:-------------:|:-------------:| 
| 560, 465      | 320, 0        | 
| 730, 465      | 960, 0	    |
| 1120, 715     | 960, 720      |
| 160, 715      | 320, 720      |

I verified that my perspective transform was working as expected by drawing the `src` and `dst` points onto a test image (see notebook) and its warped counterpart to verify that the lines appear parallel in the warped image, the latter being below:

![alt text][image4]

#### 4. Describe how (and identify where in your code) you identified lane-line pixels and fit their positions with a polynomial.

I used a histogram of the lower half of the image to get a count of positive hits, and used the peaks on either side of the centre to detect the base of the two lines. I then used windows to detect pixles within a certain distance from my starting position, updating the position of these windows whenever I found enough pixels (the code for all of this is found in 'Lane.py'). I considered trying to constrain the search, but I recognised from the harder challenge video that this would be too restrictive since really sharp turns can go right to the edges (and even off!!) the screen, so I simply tried to maximise the quality of my thresholding to avoid this.

Then when I was done having stored all of the pixels, I stored them into separate arrays and fitted them using numpy's `np.polyfit()` method. I drew these pixels detected (and the windows in which they were detected) in a method called `draw_lines()`, the results of which are below:

![alt text][image5]

#### 5. Describe how (and identify where in your code) you calculated the radius of curvature of the lane and the position of the vehicle with respect to center.

For non-video images, I did this in my function `get_radius()` in 'Lane.py'. My methodology was this. First I retrieved the pixels as above and fitted them with lines. But for the radius and vehicle position (assuming the road is a flat plane and the camera is at the centre of the car), the outputs are not in pixels, but metres. So what I had to do was store another copy of the fits by multiplying through the x and y coordinates by the corresponding conversions (similar to Udacity's ones provided). Then I performed a mathematical operation (provided at this link: http://www.intmath.com/applications-differentiation/8-radius-curvature.php). This operation took the coefficients that I applied the conversion to and produced the radii for both lines, which I then took the average of to give the radius of curvature from the bottom centre of the image. To find the centre of the vehicle, I evaluated the results of my non-converted coefficients with the y point at the bottom of the image to get two points. I then simply took the absolute distance from the centre of the average of the two lines, and multiplied the result by the conversion between pixels and metres to get the distance in metres. 

#### 6. Provide an example image of your result plotted back down onto the road such that the lane area is identified clearly.

I implemented this step in `get_radius()` of `Lane.py` as well, since it just seemed that I may as well just do the whole process if I came as far as one would in calculating the radius correctly.  Here is an example of my result on a test image:

![alt text][image6]

---

### Pipeline (video)

#### 1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (wobbly lines are ok but no catastrophic failures that would cause the car to drive off the road!).

Here's a [link to my video result](./output_video/output_video.mp4)

---

### Discussion

#### 1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

Here I'll talk about the approach I took, what techniques I used, what worked and why, where the pipeline might fail and how I might improve it if I were going to pursue this project further.  

So with regards to my approach, this was definitely the toughest project so far because the camera work is so damn fiddly. The key thing seemed to be getting a consistently good thresholding upfront, because I've seen some good work by others being totally destroyed on really tough videos like those nightmarishly hard challenge videos. That's why I ended up ignoring gradients, as these were very noisy. Perhaps a better lane-finding algorithm instead of my blind search might be better at ignoring noise.

Another really tough job was eliminating other details like car shadows. That's why my warping essentially 'stretches' the lane to encompass most of the image when retrieving pixels, since it avoids the risk of these irrelevant details being large enough to steal the peaks and mess up the algorithm.

I also have to say that the code itself was seriously hard to wrap my head around - this is the first time I needed to be able to use code collapsing (a feature that Jupyter Notebooks should really come with by default), and hence I broke up my code into separate code modules. This did help, although I confess having to restart the Jupyter kernel whenever I changed something in those modules definitely slowed development time. 

Regarding where my pipeline fails, well not at all in the project video thankfully! I couldn't get it working on the harder videos, and I believe this is because it failed to detect any lines at all in the easier video at one point, so it couldn't continue. The pipeline does assume that it can at least get some progress on each frame, which I recognise as being very hard to actually achieve. I suspect smoothing and momentum-based weighting could help to tide over the pipeline while it scrambles to start picking up lanes again.

I think my radii are somewhat off the readings Udacity got, although my method is the same so it is likely due to lack of sanity-checking. I do do some sanity checking when updating the best coefficients to map the lane lines though, which helps. They are at least *generously* in the right ballpark, so that is something.

Overall, a tough project and one I hope to improve upon eventually. I recognise a neural net might be too much for this task, but maybe that could handle the highly volatile nature of lane lines better than this approach, as I struggle to imagine how one could ever achieve a completely generalisable program with this approach. It would have to be monumentally complicated to handle every possible scenario, far more than what I have managed to achieve, and kudos to those who keep working at it.
















