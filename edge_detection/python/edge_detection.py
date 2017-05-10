from optparse import OptionParser
import cv2
import os

class EdgeDetection:
    
    def create_image_with_edges(self, input_img, output_img):
        input = cv2.imread(input_img)
        # Convert to gray scale
        img_gray = cv2.cvtColor(input, cv2.COLOR_BGR2GRAY)
        # Apply gaussian blur
        img_gray_blur = cv2.GaussianBlur(img_gray, (5,5), 0)
        # Apply canny edge detector
        canny_edges = cv2.Canny(img_gray_blur, 10, 70)
        # Apply threshold
        ret, mask = cv2.threshold(canny_edges, 70, 255, cv2.THRESH_BINARY_INV)
        cv2.imwrite(output_img, mask)
        
        

if __name__ == '__main__':

    parser = OptionParser()
    parser.add_option("-i", "--input-image", dest="input_img",
                      help="Input image", metavar="FILE")
    parser.add_option("-o", "--output-image", dest="output_img",
                      help="Output image image", metavar="FILE")

    (options, args) = parser.parse_args()

    edge_detection = EdgeDetection()
    edge_detection.create_image_with_edges(options.input_img, options.output_img)
    
    