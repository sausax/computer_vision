#include <iostream>
#include <opencv2/opencv.hpp>

using namespace std;
using namespace cv;

void drawBlobs(Mat& input,string& title, vector<KeyPoint>& keyPoints){
	// Draw keypoints
	Mat blobs;
	Scalar redColor(0, 0, 255);
	drawKeypoints(input, keyPoints, blobs, redColor);

	// Add text
	Point bottom(20, 550);
	string text = "Total blobs: " + to_string(keyPoints.size());
	cout << text;
	putText(blobs, text, bottom, FONT_HERSHEY_COMPLEX, 1, redColor, 1);
	imshow("Image with blobs", blobs);
	waitKey(0);

}

void detectBlobs(const string& inputImage){

	// Show input image
	Mat input;
	input = imread(inputImage, CV_LOAD_IMAGE_COLOR);
	imshow("Input image", input);
	waitKey(0);

	// Detect blob
	Ptr<SimpleBlobDetector> blobDetector = SimpleBlobDetector::create();
	vector<KeyPoint> keyPoints;
	blobDetector->detect(input, keyPoints, DrawMatchesFlags::DRAW_RICH_KEYPOINTS);
	string title = "Raw blobs";
	drawBlobs(input, title, keyPoints);


	//Ptr<SimpleBlobDetector::Params> params = SimpleBlobDetector::Params();

	SimpleBlobDetector::Params params;

	params.filterByArea = true;
	params.minArea = 100;

	params.filterByCircularity = true;
	params.minCircularity = 0.9;

	params.filterByConvexity = false;
	params.minConvexity = 0.2;

	params.filterByInertia = true;
	params.minInertiaRatio = 0.01;

	blobDetector = SimpleBlobDetector::create(params);
	vector<KeyPoint> circleKeyPoints;
	blobDetector->detect(input, circleKeyPoints, DrawMatchesFlags::DRAW_RICH_KEYPOINTS);
	string circleTitle = "Circular blobs";
	drawBlobs(input, circleTitle, circleKeyPoints);
}

int main(int argc, char** argv) {
	detectBlobs("../blobs.jpg");
	return 0;
}
