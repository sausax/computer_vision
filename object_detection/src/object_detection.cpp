#include <iostream>
#include <opencv2/opencv.hpp>

using namespace std;
using namespace cv;

void detectObject(Mat& testImage, Mat& templateImage){
	// Crop test image
	Rect roi(600, 400, 500, 450);
	Mat croppedImage = testImage(roi);

	Ptr<ORB> orb = ORB::create(2000, 1.2);

	// Getting features from testImage
	Mat descriptor1;
	vector<KeyPoint> keyPoint1;
	orb->detectAndCompute(croppedImage, noArray(), keyPoint1, descriptor1);

	//Getting features from template image
	Mat descriptor2;
	vector<KeyPoint> keyPoint2;
	orb->detectAndCompute(templateImage, noArray(), keyPoint2, descriptor2);

	Ptr<BFMatcher> matcher = BFMatcher::create(NORM_HAMMING, false);
	vector<DMatch> matches;
	matcher->match(descriptor1, descriptor2, matches);

	sort(matches.begin(), matches.end(), [](const DMatch& d1, const DMatch& d2){return d1.distance < d2.distance;});

	int threshold = 1700;

	Point top(50, 50);
	Scalar red(0,0,255);
	Scalar blue(255,0,0);

	if(matches.size() > threshold){
		putText(testImage, "Match found", top, FONT_HERSHEY_COMPLEX, 1, blue, 1);
	}else{
		putText(testImage, "Match not found", top, FONT_HERSHEY_COMPLEX, 1, red, 1);
	}

	imshow("Output image", testImage);
	waitKey(0);

}

int main() {
	Mat obj;
	obj = imread("../object.jpg", CV_LOAD_IMAGE_GRAYSCALE);

	Mat withObj;
	withObj = imread("../with_object.jpg", CV_LOAD_IMAGE_GRAYSCALE);

	Mat withoutObj;
	withoutObj = imread("../without_object.jpg", CV_LOAD_IMAGE_GRAYSCALE);

	detectObject(withObj, obj);
	detectObject(withoutObj, obj);
	return 0;
}
