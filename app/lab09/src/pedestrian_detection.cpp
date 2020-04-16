#include <iostream>
#include <vector>
#include "opencv2/opencv.hpp"

using namespace std;
using namespace cv;

int main()
{
	//Read Video & Classifier
	VideoCapture cap1("../data/mitsubishi_768x576.avi");
	String fullbody_name = "../data/haarcascade_fullbody.xml";
	CascadeClassifier fullbody;
	fullbody.load(fullbody_name);

	vector<Rect> faces;
	Mat gray_Image;

	if (!cap1.isOpened())
	{
		printf("not read file \n");
	}
	Mat frame1;
	namedWindow("video", 1);

	for (;;)
	{
		//read frame
		cap1 >> frame1;
		Mat original = frame1.clone();
		Mat gray;

		//Converts to a gray image & Histogram equalization
		if (frame1.channels() > 1)
		{
			cv::cvtColor(original, gray, CV_BGR2GRAY);
			cv::equalizeHist(gray, gray);
		}
		else
		{
			gray = frame1;
		}
		//Detect pedestraians
		vector<Rect> pedestraians;
		fullbody.detectMultiScale(gray, pedestraians, 1.1, 2, 0, Size(30, 30), Size(150, 150));

		for (size_t i = 0; i < pedestraians.size(); i++)
		{
			//draw ellipse
			Point center(pedestraians[i].x + pedestraians[i].width*0.5, pedestraians[i].y + pedestraians[i].height*0.5);
			ellipse(frame1, center, Size(pedestraians[i].width*0.5, pedestraians[i].height*0.5), 0, 0, 360,
				Scalar(255, 0, 255), 4, 8, 0);
		}



		imshow("video", frame1);
		if (cv::waitKey(20) == 27) break; //exit - ESCKEY
	}
	return 0;
}