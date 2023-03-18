import cv2
import numpy as np
import glob

CHECKERBOARD = (8,6)
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
 
# Creating vector to store vectors of 3D points for each checkerboard image
objpoints = []
# Creating vector to store vectors of 2D points for each checkerboard image
imgpoints = [] 
 
 
# Defining the world coordinates for 3D points
objp = np.zeros((1, CHECKERBOARD[0] * CHECKERBOARD[1], 3), np.float32)
objp[0,:,:2] = np.mgrid[0:CHECKERBOARD[0], 0:CHECKERBOARD[1]].T.reshape(-1, 2)*25
prev_img_shape = None


# Extracting path of individual image stored in a given directory
images = glob.glob('./calibration/*.png')
for fname in images:
    img = cv2.imread(fname)
    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    # Find the chess board corners
    # If desired number of corners are found in the image then ret = true
    ret, corners = cv2.findChessboardCorners(gray, CHECKERBOARD, cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_FAST_CHECK + cv2.CALIB_CB_NORMALIZE_IMAGE)
     
    """
    If desired number of corner are detected,
    we refine the pixel coordinates and display 
    them on the images of checker board
    """
    if ret == True:
        objpoints.append(objp)
        # refining pixel coordinates for given 2d points.
        corners2 = cv2.cornerSubPix(gray, corners, (5,5),(-1,-1), criteria)
         
        imgpoints.append(corners2)
 
        # Draw and display the corners
        img = cv2.drawChessboardCorners(img, CHECKERBOARD, corners2, ret)
     
    # cv2.imshow('img',img)
    # cv2.waitKey(0)
 
cv2.destroyAllWindows()
 
h,w = img.shape[:2]
 
"""
Performing camera calibration by 
passing the value of known 3D points (objpoints)
and corresponding pixel coordinates of the 
detected corners (imgpoints)
"""
ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)
 
print("Camera matrix : \n")
print(mtx)
# print("dist : \n")
# print(dist)
# print("rvecs : \n")
# print(rvecs)
# print("tvecs : \n")
# print(tvecs)

K = mtx

Rt = []

for rvec, tvec in zip(rvecs, tvecs):
    R_matrix, _ = cv2.Rodrigues(rvec)

    Rt_matirx = np.concatenate((R_matrix, tvec), axis=1)
    Rt.append(Rt_matirx)

Rt = np.array(Rt)
# print('Rt: \n', Rt)


# import the necessary packages
from imutils import paths
import numpy as np
import imutils
import cv2

def find_marker(image):
	# convert the image to hsv, threshold it, blur it, and detect edges
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    img_thresh_low = cv2.inRange(hsv, np.array([0, 135, 135]), np.array([15, 255, 255])) 
    img_thresh_high = cv2.inRange(hsv, np.array([159, 135, 135]), np.array([179, 255, 255]))
    img_thresh = cv2.bitwise_or(img_thresh_low, img_thresh_high)

    kernel = np.ones((5, 5))
    img_thresh_opened = cv2.morphologyEx(img_thresh, cv2.MORPH_OPEN, kernel)

    img_thresh_blurred = cv2.medianBlur(img_thresh_opened, 5)

    img_edges = cv2.Canny(img_thresh_blurred, 80, 160)

    contours, _ = cv2.findContours(np.array(img_edges), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE) 


    approx_contours = []

    for c in contours:
        approx = cv2.approxPolyDP(c, 10, closed = True)
        approx_contours.append(approx)

    all_convex_hulls = []
    for ac in approx_contours:
        all_convex_hulls.append(cv2.convexHull(ac))

    c = all_convex_hulls

    c = max(c, key = cv2.contourArea)
    # print('c: ', c)
    c = cv2.minAreaRect(c)
	# compute the bounding box of the of the paper region and return it
    return c

def distance_to_camera(knownWidth, focalLength, perWidth):
	# compute and return the distance from the maker to the camera
	return (knownWidth * focalLength) / perWidth


KNOWN_DISTANCE = 40.0
KNOWN_WIDTH = 40.0

image = cv2.imread("resource/cone_x40cm.png")
# marker = find_marker(image)
# focalLength = (marker[1][0] * KNOWN_DISTANCE) / KNOWN_WIDTH
focalLength = (K[0,0] + K[1,1])/2
# print(focalLength)

image = cv2.imread("resource/cone_x40cm.png")
marker = find_marker(image)

dist = distance_to_camera(KNOWN_WIDTH, focalLength, marker[1][0])
focalLength = focalLength*KNOWN_DISTANCE/dist # recalibration to current case
dist = distance_to_camera(KNOWN_WIDTH, focalLength, marker[1][0])
# draw a bounding box around the image and display it
box = cv2.cv.BoxPoints(marker) if imutils.is_cv2() else cv2.boxPoints(marker)
box = np.int0(box)
cv2.drawContours(image, [box], -1, (0, 255, 0), 2)
cv2.putText(image, "%.2fcm" % (dist),
    (image.shape[1] - 200, image.shape[0] - 20), cv2.FONT_HERSHEY_SIMPLEX,
    1.0, (0, 255, 0), 3)
cv2.imshow("image - 40", image)
cv2.waitKey(0)


image = cv2.imread("resource/cone_unknown.png")
marker = find_marker(image)

dist = distance_to_camera(KNOWN_WIDTH, focalLength, marker[1][0])
# draw a bounding box around the image and display it
print('marker: ', marker)
box = cv2.cv.BoxPoints(marker) if imutils.is_cv2() else cv2.boxPoints(marker)
box = np.int0(box)
cv2.drawContours(image, [box], -1, (0, 255, 0), 2)
cv2.putText(image, "%.2fcm" % (dist),
    (image.shape[1] - 200, image.shape[0] - 20), cv2.FONT_HERSHEY_SIMPLEX,
    1.0, (0, 255, 0), 3)
cv2.imshow("image", image)
cv2.waitKey(0)

