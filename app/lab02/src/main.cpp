#include <iostream>
#include <opencv.hpp>
//#include <core.hpp>
//#include <highgui.hpp>
//#include <imgproc.hpp>
#include <stdio.h>

using namespace cv;
using namespace std;

#define MORPH 1
#define DISSOLVE 0

int main()
{
#if MORPH //Morphological filtering
	Mat original = imread("../data/number2.jpg", 0);

	Mat binary_image;
	int th = 127;
	threshold(original, binary_image, th, 255, THRESH_BINARY);
	namedWindow("original");
	imshow("Image", original);
	waitKey(0);

	Mat eroded_image;
	Mat element = getStructuringElement(MORPH_CROSS, Size(3, 3), Point(1, 1));
	erode(binary_image, eroded_image, Mat());
	namedWindow("eroded image");
	imshow("eroded image", eroded_image);
	waitKey(0);

	imwrite("../data/eroded.jpg", eroded_image);

	Mat dilated_image;
	element = getStructuringElement(MORPH_RECT, Size(3, 3), Point(1, 1));
	dilate(binary_image, dilated_image, element);
	namedWindow("dilated image");
	imshow("dilated image", dilated_image);

	waitKey(0);

	imwrite("../data/dilated.jpg", dilated_image);

	Mat closed;
	morphologyEx(binary_image, closed, MORPH_CLOSE, element);
	imshow("closed image", closed);

	waitKey(0);

	imwrite("../data/closed.jpg", closed);

	Mat opened;
	morphologyEx(binary_image, opened, MORPH_OPEN, element);
	imshow("opened image", opened);
	
	waitKey(0);

	imwrite("../data/opened.jpg", opened);

	return 0;
#endif
#if DISSOLVE 
	Mat src1 = imread("../data/deer.jpg", CV_LOAD_IMAGE_COLOR);
	Mat src2 = imread("../data/catdog.jpg", CV_LOAD_IMAGE_COLOR);
	Mat dst;

	int value;

	resize(src1, src1, Size(320, 240));
	resize(src2, src2, Size(320, 240));

	namedWindow("Dissolving", CV_WINDOW_AUTOSIZE);

	createTrackbar("È¥ÇÕ°è¼ö", "Dissolving", &value, 100);
	setTrackbarPos("È¥ÇÕ°è¼ö", "Dissolving", 80);

	while (true) {

		value = getTrackbarPos("È¥ÇÕ°è¼ö", "Dissolving");
		double alpha = (value / 100.0);
		double beta = 1 - alpha;

		addWeighted(src1, alpha, src2, beta, 0.0, dst);
		imshow("Dissolving", dst);

		if (waitKey(1) == 'q') break;
	}

	waitKey(0);
	return 0;
#endif

}