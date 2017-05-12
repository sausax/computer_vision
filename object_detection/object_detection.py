import cv2
import numpy as np

def orb_detector(test_image, template_image):
    cropped_image = test_image[400:750, 600:1100]
    gray_img = cv2.cvtColor(cropped_image, cv2.COLOR_BGR2GRAY)
    gray_template = cv2.cvtColor(template_image, cv2.COLOR_BGR2GRAY)
    
    # Create ORB detector with 1000 keypoints with a scaling pyramid factor of 1.2
    orb = cv2.ORB_create(1000, 1.2)
    
    
    keypoint1, descriptors1 = orb.detectAndCompute(gray_img, False)
    keypoint2, descriptors2 = orb.detectAndCompute(gray_template, False)
    
    # Create matcher
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    
    matches = bf.match(descriptors1, descriptors2)

    # Sort the matches based on distance.  Least distance
    # is better
    matches = sorted(matches, key=lambda val: val.distance)
    
    threshold = 114
    print len(matches)
    
    # If matches exceed our threshold then object has been detected
    if len(matches) > threshold:
        cv2.rectangle(test_image, (600,400), (1100,750), (0,255,0), 3)
        cv2.putText(test_image,'Object Found',(50,50), cv2.FONT_HERSHEY_COMPLEX, 2 ,(0,255,0), 2)
    else:
        cv2.rectangle(test_image, (600,400), (1100,750), (0,0,255), 3)
        cv2.putText(test_image,'Object Not Found',(50,50), cv2.FONT_HERSHEY_COMPLEX, 2 ,(0,0,255), 2)
    
    cv2.imshow('Object Detector using ORB', test_image)
    cv2.waitKey(0)


def detect_object(object_img, with_object_img, without_object_img):
    obj = cv2.imread(object_img)
    with_obj = cv2.imread(with_object_img)
    print with_obj.shape
    without_obj = cv2.imread(without_object_img)
    
    orb_detector(with_obj, obj)
    orb_detector(without_obj, obj)
    

if __name__ == '__main__':
    object_img = 'object.jpg'
    with_object_img = 'with_object.jpg'
    without_object_img = 'without_object.jpg'
    detect_object(object_img, with_object_img, without_object_img)