import numpy as np
import cv2


def find_marker(image):
	# convert the image to hsv, threshold it, blur it, and detect edges
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    # cv2.imshow('hsv', hsv)
    # cv2.waitKey(0)

    img_thresh_low = cv2.inRange(hsv, np.array([20, 100, 100]), np.array([30, 255, 255])) 
    img_thresh_high = cv2.inRange(hsv, np.array([25, 50, 70]), np.array([80, 255, 255]))
    img_thresh = cv2.bitwise_or(img_thresh_low, img_thresh_high)
    # img_thresh = img_thresh_low

    kernel = np.ones((5, 5))
    img_thresh_opened = cv2.morphologyEx(img_thresh, cv2.MORPH_OPEN, kernel)

    img_thresh_blurred = cv2.medianBlur(img_thresh_opened, 5)

    img_edges = cv2.Canny(img_thresh_blurred, 80, 160)

    contours, _ = cv2.findContours(np.array(img_edges), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE) 

    cv2.drawContours(image, contours, -1, (255,255,255), 2)
    cv2.imshow('contours', image)
    cv2.waitKey(0)

    c = contours

    c = [cv2.minAreaRect(i) for i in c]

    return c


def find_lane(image):
    
    markers = find_marker(image)

    for marker in markers:
        box = cv2.boxPoints(marker)
        box = np.int0(box)
        cv2.drawContours(image, [box], -1, (0, 255, 0), 2)
    cv2.imshow("lane markings", image)
    cv2.waitKey(0)

if __name__ == '__main__':
    image = cv2.imread("resource/lane.png")
    find_lane(image)




