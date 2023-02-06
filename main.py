import numpy as np
import cv2
import math


# Detect whether the feature is the background every 30 frames.
# If it is the background, delete the feature point
# Also delete the color label
def detect_background(fgbg, frame_f, good_new_f, good_old_f, my_Kmean_label_f):
    # Apply the Background Subtraction
    frame_fgbg = fgbg.apply(frame_f)
    count1 = 0
    length_old = good_new_f.shape[0]
    # Judge whether the feature point is the background
    while count1 < length_old:
        if frame_fgbg[int(good_new_f[count1][1]), int(good_new_f[count1][0])] == 0:
            good_old_f = np.delete(good_old_f, count1, 0)
            good_new_f = np.delete(good_new_f, count1, 0)
            # Delete feature points in the background
            my_Kmean_label_f = np.delete(my_Kmean_label_f, count1)
            my_Kmean_label_f = my_Kmean_label_f.reshape(-1, 1)
            length_old = good_new_f.shape[0]
        else:
            count1 += 1
    # Returns the new feature point ndarrays and color list
    return [good_new_f, good_old_f, my_Kmean_label_f]


# K-means algorithm is used to classify feature points
# Get my_Kmean_label_F as color list
def kmean_cluster(p0_f):
    my_Kmean_k = 2
    my_Kmean_f = p0_f[:, 0, :]
    my_Kmean_z_f = np.float32(my_Kmean_f)
    criteria_f = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
    # Do the K-means classification
    my_Kmean_ret_f, my_Kmean_label_f, my_Kmean_center_f = \
        cv2.kmeans(my_Kmean_z_f, my_Kmean_k, None, criteria_f, 10, cv2.KMEANS_RANDOM_CENTERS)
    return [my_Kmean_ret_f, my_Kmean_label_f, my_Kmean_center_f]


# Canny algorithm is used for edge detection
# Returns an image with red lines drawing edges
def my_canny(my_frame):
    # Change the color of the edge in red
    lam = [0.01, 0.01, 1.0]
    # Perform the Canny algorithm
    canny = cv2.Canny(my_frame, 120, 180)
    canny_mask = cv2.cvtColor(canny, cv2.COLOR_GRAY2BGR)
    canny_mask = canny_mask * lam
    canny_mask = np.uint8(canny_mask)
    img = cv2.add(my_frame, canny_mask)
    return img


# Mouse callback function
# Change the global variables IX and iy to the current mouse position
def draw_circle(event,x,y,flags,param):
    global ix, iy, changed
    if event == cv2.EVENT_LBUTTONDOWN:
        changed = True
        ix = x
        iy = y


# params for ShiTomasi corner detection
feature_params = dict(maxCorners=100,
                      qualityLevel=0.1,
                      minDistance=7,
                      blockSize=7)

# Parameters for lucas kanade optical flow
lk_params = dict(winSize=(18, 18),
                 maxLevel=2,
                 criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.4))

# Check the mouse callback change or not
changed = False
# Read the Video
cap = cv2.VideoCapture("input.mov")
# get the video fps
fps = cap.get(cv2.CAP_PROP_FPS)
# get the size of video
size = (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
# change the format of output video
fourcc = cv2.VideoWriter_fourcc(*'XVID')
# set the output video
out = cv2.VideoWriter('output.avi', fourcc, fps, size)

# create the Background Subtractor
fgbg = cv2.createBackgroundSubtractorKNN()
# Create some random colors
color = np.random.randint(0, 255, (100, 3))

# Take first frame and find corners in it
ret, old_frame = cap.read()
# convert the BGR into gray
old_gray = cv2.cvtColor(old_frame, cv2.COLOR_BGR2GRAY)
# Perform the ShiTomasi corner detection
p0 = cv2.goodFeaturesToTrack(old_gray, mask=None, **feature_params)

# Create a mask image for drawing purposes
mask = np.zeros_like(old_frame)

# Set the frame count
frame_count = 0
# Do the K means cluster
my_Kmean_ret, my_Kmean_label, my_Kmean_center = kmean_cluster(p0)

# Read the video
while 1:
    ret, frame = cap.read()
    # If the video has the frame, do the operation
    # Else break the loop
    if ret:
        # Convert the BGR color into gray
        frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Avoid could not find the goodFeatures in the first frame
        if p0 is None:
            p0 = cv2.goodFeaturesToTrack(frame_gray, mask=None, **feature_params)
            old_gray = frame_gray
            continue

        # calculate optical flow
        p1, st, err = cv2.calcOpticalFlowPyrLK(old_gray, frame_gray, p0, None, **lk_params)

        # Avoid could not find the goodFeatures in the new frame
        if p1 is None:
            p0 = cv2.goodFeaturesToTrack(frame_gray, mask=None, **feature_params)
            old_gray = frame_gray
            continue

        # Select good points
        good_old = p0[st == 1]
        good_new = p1[st == 1]

        # Detect whether the feature is the background every 30 frames.
        # If it is the background, delete the feature
        if math.fmod(frame_count, 30) == 0 and frame_count > 30:
            [good_new, good_old, my_Kmean_label] = detect_background(fgbg, frame, good_new, good_old, my_Kmean_label)

        # Draw the tracks
        for i, (new, old) in enumerate(zip(good_new, good_old)):
            # Get to position of the feature
            a, b = new.ravel()
            c, d = old.ravel()
            # Convert the position as int
            a = int(a)
            b = int(b)
            c = int(c)
            d = int(d)
            # Draw the Different kinds of feature point in different color
            # Category 1
            if my_Kmean_label[i] == 0:
                frame = cv2.circle(frame, (a, b), 5, (255, 0, 0), -1)
                mask = cv2.line(mask, (a, b), (c, d), (255, 0, 255), 2)
            # Category 2
            elif my_Kmean_label[i] == 1:
                frame = cv2.circle(frame, (a, b), 5, (0, 0, 255), -1)
                mask = cv2.line(mask, (a, b), (c, d), (0, 255, 255), 2)
            # The point with newly insert
            else:
                frame = cv2.circle(frame, (a, b), 5, (0, 255, 0), -1)
                mask = cv2.line(mask, (a, b), (c, d), (255, 255, 0), 2)
        # Draw the Line
        img = cv2.add(frame, mask)
        # Do the edge detection
        img = my_canny(img)

        # Set the Mouse call back
        cv2.namedWindow('ooo')
        cv2.setMouseCallback('ooo', draw_circle)
        # Show the image
        cv2.imshow('ooo', img)
        # Write the frame into out
        out.write(img)
        # Press q to quit the program
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

        # Update the previous frame and previous points
        old_gray = frame_gray.copy()
        # If users point the img
        # Add the new feature point
        if changed:
            insert_point = np.array([ix, iy])
            good_new = np.vstack((good_new, insert_point))
            my_Kmean_label = np.vstack((my_Kmean_label, [2]))
            changed = False
        # Reset the p0
        p0 = good_new.reshape(-1, 1, 2)
        p0 = np.float32(p0)
        # Add frame count
        frame_count += 1
    else:
        break

# Destroy the windows and release
cv2.destroyAllWindows()
cap.release()
out.release()
