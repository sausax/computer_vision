//============================================================================
// Name        : shape_detection.cpp
// Author      : 
// Version     :
// Copyright   : Your copyright notice
// Description : Hello World in C++, Ansi-style
//============================================================================

#include <iostream>
#include <string>
#include <opencv2/opencv.hpp>

using namespace std;
using namespace cv;

void printShape(Mat& image, vector<Point>& contour, string& name, Scalar& color){
	vector<vector<Point>> contours;
	contours.push_back(contour);
	drawContours(image,contours,0,color,-1);
	Moments moment = moments(contour, false);
	int cx = int(moment.m10/moment.m00);
	int cy = int(moment.m01/moment.m00);
	Point center(cx-60, cy);
	Scalar black(0, 0, 0);
	putText(image, name, center, FONT_HERSHEY_COMPLEX, 1, black, 1);
}

void detectShape(const string& inputFile, const string& outputFile){
	// Reading input image file
	Mat input;
	input = imread(inputFile, CV_LOAD_IMAGE_COLOR);

	imshow("Input Image", input);
	waitKey(0);

	// Convert image to gray scale
	Mat grayImage;
	cvtColor(input, grayImage, COLOR_BGR2GRAY);

	// Get threshold value from image
	Mat thresholdImage;
	threshold(grayImage, thresholdImage, 70, 255, THRESH_BINARY);

	// Get contours
	vector<vector<Point> > contours;
	findContours(thresholdImage, contours, RETR_LIST, CHAIN_APPROX_NONE);

	//Remove top level bounding contour
	sort(contours.begin(), contours.end(),
			[](const vector<Point>& c1, const vector<Point>& c2){
				return contourArea(c1, false) < contourArea(c2, false);
			}
	);
	contours.pop_back();

	// Iterate over contours
	for(auto& contour: contours){
		// Detect closest polygon
		vector<Point> polySides;
		double epsilon = 0.01 * arcLength(contour, true);
		approxPolyDP(contour, polySides, epsilon, true);

		// Label image with polygon name
		int sides = polySides.size();
		if(sides == 3){
			cout << "Found a triangle";
			Scalar color(255,0,0);
			string name = "Triangle";
			printShape(input, contour, name, color);
		}else if(sides == 4){
			Rect rect = boundingRect(contour);
			if(abs(rect.width - rect.height) <= 3){
				string name = "Square";
				Scalar color(0,255,0);
				printShape(input, contour, name, color);
			}else{
				string name = "Rectangle";
				Scalar color(0,0,255);
				printShape(input, contour, name, color);
			}
		}else if(sides == 10){
			string name = "Star";
			Scalar color(255,255,0);
			printShape(input, contour, name, color);
		}else if(sides >= 15){
			string name = "Circle";
			Scalar color(0, 255,255);
			printShape(input, contour, name, color);
		}
	}

	// Save output image
	imwrite(outputFile, input);
}

int main(int argc, char** argv) {
	string inputFile = "../someshapes.jpg";
	string outputFile = "../labeled_shapes.jpg";

	detectShape(inputFile, outputFile);

	return 0;
}
