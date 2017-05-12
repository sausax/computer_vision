import cv2
import os
import numpy as np

from optparse import OptionParser

def detect_blob(input_img):
    # Read image file
    input = cv2.imread(input_img)
    cv2.imshow("Input image", input)
    cv2.waitKey(0)
    
    # Create blob detector object
    detector = cv2.SimpleBlobDetector_create()
    # Detect blob
    keypoints = detector.detect(input)
    # Draw blobs as red circle
    blank = np.zeros((1,1))
    blobs = cv2.drawKeypoints(input, keypoints, blank, (0,0,255), cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
    # Add description to image
    num_of_blobs = len(keypoints)
    text = "Number of blobs: "+str(num_of_blobs)
    cv2.putText(blobs, text, (20, 550), cv2.FONT_HERSHEY_SIMPLEX, 1, (100, 0, 255), 2)
    cv2.imshow("Blobs using default parameters", blobs)
    cv2.waitKey(0)

    # Create a new blob detector to detect circles
    params = cv2.SimpleBlobDetector_Params()
    
    # Set Area filtering parameters
    params.filterByArea = True
    params.minArea = 100

    # Set Circularity filtering parameters
    params.filterByCircularity = True 
    params.minCircularity = 0.9

    # Set Convexity filtering parameters
    params.filterByConvexity = False
    params.minConvexity = 0.2
        
    # Set inertia filtering parameters
    params.filterByInertia = True
    params.minInertiaRatio = 0.01

    # Create a detector with the parameters
    detector = cv2.SimpleBlobDetector_create(params)
    
    # Detect blob
    keypoints = detector.detect(input)
    # Draw blobs as green circle
    blank = np.zeros((1,1))
    blobs = cv2.drawKeypoints(input, keypoints, blank, (0,255,0), cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
    # Add description to image
    num_of_blobs = len(keypoints)
    text = "Number of circles: "+str(num_of_blobs)
    cv2.putText(blobs, text, (20, 550), cv2.FONT_HERSHEY_SIMPLEX, 1, (100, 0, 255), 2)
    cv2.imshow("Blobs using default parameters", blobs)
    cv2.waitKey(0)

if __name__ == '__main__':
    detect_blob("blobs.jpg")