# Computer_vision_object_tracking_optical_flow
## Introduction

The primary purpose of this project is to perform object tracking in a given video by using sparse optical flow. Good feature points are found by the feature point detection method (such as SIFT, SURF and Shi-Tomasi Corner Detector) and the optical flow of these points is calculated to track their position in the next frame of the video. The input of the program will be a video taken by myself, and the output will be a video displaying the track of the feature points, including the current position (track point) and the historical position (track line).  Additionally four new features have been added to my project:

1. Clustering and classification of feature points using the K-mean algorithm, represented by different coloured tracking points and tracking lines. 
2. Detects whether feature points are on moving objects background subtraction.
3. Using the Canny algorithm for object edge detection.
4. Human-computer interaction to add new tracking points by clicking on the image.

## Key Features Description and Implement

### Project Prerequisites

##### Library requirements

1. cv2 library: Perform operations such as optical flow, background subtraction, k-mean classify and edge detection on video.
2. numpy library: Perform operations on arrays or matrices.
3. math library: Do mathematical operations.

##### Input video requirements

1. Brightness is constant.
2. Motion is small
3. Points move like their neighbors

### Implementation of the sparse optical flow tracking

Firstly, the input video is obtained using the cv2.VideoCapture() function in the cv2 library. The first frame of the video is acquired and the Shi-Tomasi Corner Detector algorithm is used to perform feature detection on this frame.  cv2 provides the cv2.goodFeaturesToTrack() function to implement the Shi-Tomasi Corner Detector algorithm. WAITTOADD Also users can adjust the results of feature detection by adjusting the feature_params parameter of the cv2.goodFeaturesToTrack() function. Afterwards I stored all the feature points obtained in the ndarray p0. When the above has been done, the next frame of the video is obtained by looping. I chose to use the Lucas-Kanade algorithm to calculate the sparse optical flow. WAITTOADD  cv2 library provides a highly efficient function cv2.calcOpticalFlowPyrLK() to implement the Lucas-Kanade Optical Flow algorithm. The inputs are the feature points of the previous frame p0, the grey scale map of the previous frame, and the grey scale map of this frame. The output is ndarray p1 which contains the new position of the calculated feature points in the next frame. Then, drawing the tracing lines and tracing point at the current frame, connecting the old feature points with the new ones using the cv2.line() function, i.e. tracing lines, and drawing circles at the new feature points using the cv2.circle() function, i.e. tracing points. Loop through the above operations until the last frame of the video is read. Calling this function is much more efficient than implementing Lucas-Kanade algorithm ourselves. Users can also adjust the calculation of the algorithm by adjusting the lk_params of the cv2.calcOpticalFlowPyrLK() function.



![image](https://user-images.githubusercontent.com/39216716/216953110-4f395867-8d57-4af1-b4e9-cf95104118ea.png)


### Implementation of K-means algorithm for feature pointclassification

The feature points obtained using the Feature Detection algorithm are disordered. I chose to use the K-mean algorithm to perform unsupervised machine learning on the feature points for cluster classification. K represents the number of classifications, and by default, the value of K is 2, i.e. the feature points are divided into two different categories. cv2 library provides the c2.kmeans() function to implement the K-mean algorithm, by changing the parameter K of c2.kmeans() to change the number of categories of the classification. Feature points in the same category are represented by the same colour. The track point for category \romannumeral1 is red and the track line is yellow, and the track point for category \romannumeral2 is blue and the track line is purple.

### Implementation of the Canny algorithm for edge detection

In a cluttered background sometimes it can be difficult to find our objects. Therefore I have chosen to use edge detection to make our tracked objects more visible. The Canny algorithm is an effective method that can detect the edges of an object. I have applied the Canny algorithm in my project for the purpose of edge detection. cv2 library provides an efficient implementation of the Canny algorithm with the function cv2.Canny(). cv2.Canny() takes in a frame that has been converted to a greyscale map and outputs an edge map with white edges and the rest of the frame in black. By adjusting the parameters of the cv2.Canny() function will adjust the amount of detail contained in the frame. After getting the edge map, I need to superimpose the edge map onto our original frame. Here I transformed the edge map into a three channel map and then zoomed in on the three channels (to make the edges red) and used the cv2.add() function to superimpose the processed edge map on the original image to get the effect I was looking for.

### Implementation of Background Subtraction to detect the plausibility of feature points

In some cases, during feature detection or sparse optical flow operations, it may happen that feature points are recognised in the background. Such feature points are not meaningful for the object we are tracking. Therefore I need to eliminate these useless features. Here I choose to use the Background Subtraction method to check the reasonableness of the feature points. The cv2 library provides the cv2.createBackgroundSubtractorKNN() function to perform Background Subtraction. cv2.createBackgroundSubtractorKNN() uses the KNN algorithm, where the input is an image that has been converted to a greyscale map The output is a grayscale image with the background removed, which background will be represented in black.

Every thirty frames, the program will do the background subtraction for the corresponding frame. Afterwards, the feature points are compared to the processed grayscale image. If the value of the pixel corresponding to a feature point in the grey scale map is 0, it means that the feature point is in the background and will be removed. The reason for choosing to do this operation every thirty frames instead of every frame is that doing the background subtraction every frame would cause the calculation too slow.

### Implementation of human-computer interaction to add tracking points manually

As the initial feature points are automatically generated by the feature detection algorithm, it is not possible to manually add feature points to perform sparse optical flow tracking on the user's interesting feature points. Therefore I have added human-computer interaction to this program. The user can add a new feature point by clicking the left mouse click, the newly added feature point has green track point and cyan track line. The cv2 library provides a function cv2.setMouseCallback() that gets the state of the mouse. I call this function to achieve the human-computer interaction I expect. When the user clicks on the image, the function will get the x and y coordinates of the current mouse click state via event and will add the x and y coordinates to the set of feature points. If the feature point is on a moving object, the program will track the feature point. Clicking on the background does not add a valid feature point. 

## Results Evaluation and Improvement

### Result of Test Video

By running the program, we can obtain a test video. At the beginning of the video, a series of feature points are obtained by the feature detection algorithm. Using the optical flow methodology to calculate the future position of these feature points, the object can be tracked. The video indicates the historical position of the feature points by tracking lines and the current position of the feature points by tracking points. By default, the feature points will be divided into two categories by the K-means algorithm and represented by different coloured tracking points and tracking lines. In addition, the edges of the objects in the video are drawn with red lines through the Canny algorithm in order to allow the user to distinguish the objects more clearly. A background subtraction operation is performed every thirty frames of the video, removing the feature points that are in the background. Finally by clicking on objects in the image, new feature points can be added and they can be tracked using specially coloured tracking points and tracking lines and the output video will store as "output.avi".
![image](https://user-images.githubusercontent.com/39216716/216953039-d949cd73-d363-4ace-afca-46ca1be50edd.png)


### Strengths & Weaknesses

Strengths of the project:

1. The tracked points are not lost in most cases.
2. Good compatibility with larger resolution videos.
3. Initially generated feature points are exactly on the tracked object.
4. Allows the user to add feature points manually.
5. The project has a self-correcting function that removes unnecessary feature points when they are on the background and not on the object to be tracked.
6. Different colours are used to indicate different categories of feature points.
7. Edge detection allows for a more intuitive view of the object.

Weaknesses of the project:

1. Feature points are lost when the object is moved too much.
2. Background subtraction is not precise enough and sometimes incorrectly removes the feature points that should be retained.
3. There is a delay when adding features manually.

### Evaluation of the project

1. The feature detection method chosen in the project

   At the beginning of the project I used the SIFT (Scale-Invariant Feature Transform) algorithm to implement feature detection, but I found that SIFT did not work well and many feature points were useless. The research from Navid et.al. shows that the SIFT algorithm does not perform well in terms of computation time and success tracking rate of feature points when implementing the sparse optical flow method of tracking. The specific test results for each feature detection algorithm are shown in the table below.

   I found that GFTT (Shi-Tomasi Corner Detector algorithm) has the highest success tracking rate and is also faster than  SIFT. Therefore I abandoned the SIFT algorithm and used the Shi-Tomasi Corner Detector algorithm to implement feature detection.

2. Lucas-Kanade method Evaluation

   Optimise sparse optical flow tracking of objects by adjusting the parameters of cv2.calcOpticalFlowPyrLK(). Using an image pyramid will make tracking more accurate. The Image pyramid level is changed by adjusting the parameter maxLevel. If maxLevel is too small then larger moving feature points will be lost, but if maxLevel is too large the feature points will jump to anomalous positions. Through continuous testing, I finally chose to set maxLevel to 2. The winSize parameter indicates the search window size for each pyramid level. Too large a winSize can cause a decrease in tracking accuracy. I finally chose to set winSize to (18, 18).

3. Background Subtraction Evaluation

   To improve the accuracy of Background Subtraction, I tried using different kinds of Background Subtraction methods, including KNN, cv2.createBackgroundSubtractorMOG() and cv2. BackgroundSubtractorMOG2(). By comparison, the edges of the Background Subtraction process using the KNN algorithm are smoother than the other two methods. In addition to this, I also erode and then dilate the grayscale image to remove any undesirable noise.

### Issues to enhance and future work

For complex scenes, feature points can be lost or exceptionally jumped with errors. In the future I need to increase the robotic of the algorithm and to achieve this I will read more literature to learn more about advanced sparse optical flow tracking algorithms and implement it myself.

When a scene changes significantly, then all feature points will be invalidated. In the future I need to add a new feature to my program to create new feature points and track them when a large change in the scene is detected.

In the current version of the program the user can only add feature points manually and is not allowed to manually delete specified feature points. In the future I will be implementing a new feature to acquire a region by dragging and dropping the mouse and to delete feature points from this region.
