from optparse import OptionParser
import cv2
import os
from numpy import poly

# Apply contour over shape, add color and text
def draw_shape(image, contour, shape_name, color):
    cv2.drawContours(image,[contour],0,color,-1)
     # Find contour center to place text at the center
    M = cv2.moments(contour)
    cx = int(M['m10'] / M['m00'])
    cy = int(M['m01'] / M['m00'])
    cv2.putText(image, shape_name, (cx-50, cy), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 1)

def detect_shapes(input_file, output_file):
    # Read image file
    image = cv2.imread(input_file)
    # Convert image to grayscale
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # Apply threshold to image
    ret, threshold_image = cv2.threshold(gray_image, 127, 255, cv2.THRESH_BINARY)
    # Get contours from image
    _, contours, hierarcy = cv2.findContours(threshold_image, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
    
    # Removing top level contour
    n = len(contours) - 1
    contours = sorted(contours, key=cv2.contourArea, reverse=False)[:n]
    
    for contour in contours:
        # Get closest polygon matching countour
        epsilon = 0.01 * cv2.arcLength(contour, True)
        poly_sides = cv2.approxPolyDP(contour, epsilon, True)
       
        # Count number of sides of polygon and decide shape
        if len(poly_sides) == 3:
            draw_shape(image, contour, "Triangle", (255, 0, 0))
        elif len(poly_sides) == 4:
            x,y,w,h = cv2.boundingRect(contour)
            if abs(w-h) <=3:
                draw_shape(image, contour, "Square", (0,255,0))
            else:
                draw_shape(image, contour, "Rectangle", (0,0,255))
        elif len(poly_sides) == 10:
            draw_shape(image, contour, "Star", (255,255,0))
        elif len(poly_sides) >= 15:
            draw_shape(image, contour, "Circle", (0,255,255))
        
    cv2.imwrite(output_file, image)

if __name__ == '__main__':
    input_file = 'someshapes.jpg'
    output_file = 'labeled_shapes.jpg'
    detect_shapes(input_file, output_file)