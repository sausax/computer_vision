//============================================================================
// Name        : edge_detection.cpp
// Author      : 
// Version     :
// Copyright   : Your copyright notice
// Description : Hello World in C++, Ansi-style
//============================================================================

#include <iostream>
#include <opencv2/opencv.hpp>

using namespace std;
using namespace cv;


int main( int argc, char** argv )
{
    if( argc != 3)
    {
     cout <<" Usage: ./edge_detection <input_image> <output_image>" << endl;
     return -1;
    }

    Mat image;
    image = imread(argv[1], CV_LOAD_IMAGE_COLOR);   // Read the file

    if(! image.data )                              // Check for invalid input
    {
        cout <<  "Could not open or find the image" << std::endl ;
        return -1;
    }

    namedWindow( "Display window", WINDOW_AUTOSIZE );// Create a window for display.
    imshow( "Input Image", image );                   // Show our image inside it.
    waitKey(0);                                          // Wait for a keystroke in the window

    // Convert to gray scale
    Mat grayImage;
    cvtColor(image, grayImage, CV_BGR2GRAY);
    imshow("Gray output image", grayImage );
    waitKey(0);

    // Add gaussian blur
    Mat blurImage;
    Size size(5,5);
    GaussianBlur(grayImage, blurImage, size, 0);
    imshow("Blur output image", blurImage);
    waitKey(0);

    // Detect boundaries using canny detector
    Mat cannyImage;
    Canny(blurImage, cannyImage, 10, 70);
    imshow("Canny output image", cannyImage);
    waitKey(0);

    // Apply threshold
    Mat outputImage;
    threshold(cannyImage, outputImage, 70, 255, THRESH_BINARY_INV);
    imshow("Final output image", outputImage);
    waitKey(0);

    cout << "Writing to " << argv[2] << endl;
    imwrite(argv[2], outputImage);
    // Save final image in output image file

    return 0;
}
