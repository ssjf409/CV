#include <opencv.hpp>

using namespace std;
using namespace cv;

#define EX1 0
#define EX2 0
#define EX3 0
#define EX4 0
#define EX5 0
#define EX6 1

int main()
{
#if EX1
	namedWindow("Hello, OpenCV", WINDOW_AUTOSIZE);

	Mat image = Mat::zeros(480, 640, CV_8UC3);

	cout << "image size: " << image.cols << " x " << image.rows << endl;

	putText(image, "Hello, OpenCV", Point(0, image.rows / 2), FONT_HERSHEY_PLAIN, 3, Scalar(0, 255, 0), 4);

	imshow("Hello, OpenCV", image);

	waitKey(10000);

	return 0;
#endif

#if EX2
	//namedWindow("lena", WINDOW_AUTOSIZE);

	Mat image = imread("../data/lena.png");
	Mat resized_image;

	resize(image, resized_image, Size(256, 512));

	imshow("lena", image);
	imshow("resized lena", resized_image);

	imwrite("../data/lena_resized.png", resized_image);

	waitKey(0);

	return 0;
#endif

#if EX3
	namedWindow("lena", WINDOW_AUTOSIZE);

	Mat image = imread("../data/lena.png");

	putText(image, "lena", Point(image.cols * 0.33, image.rows * 0.2), FONT_HERSHEY_PLAIN, 3, Scalar(0, 255, 0), 4);

	imshow("lena", image);

	waitKey(0);

	return 0;
#endif

#if EX4
	Mat image = Mat::zeros(480, 640, CV_8UC3);

	int x1 = image.cols / 2 - 100;
	int y1 = image.rows / 2 - 100;
	int x2 = image.cols / 2 + 100;
	int y2 = image.rows / 2 + 100;

	Point center = Point(0.5*(x1 + x2), 0.5*(y1 + y2));
	int radius = center.x - x1;

	rectangle(image, Point(x1, y1), Point(x2, y2), Scalar(0, 255, 255), 4);

	circle(image, center, radius, Scalar(100, 100, 200), 4);

	line(image, Point(x1, y1), Point(x2, y2), Scalar(255, 0, 0), 4);

	imshow("Hello, OpenCV", image);

	waitKey(0);

	return 0;
#endif

#if EX5
	Mat image = imread("../data/lena.png");
	Mat gray_image;
	cvtColor(image, gray_image, CV_BGR2GRAY);

	imshow("gray lena", gray_image);

	Mat red, green, blue;
	image.copyTo(red);
	image.copyTo(green);
	image.copyTo(blue);

	for (int y = 0; y < image.rows; ++y)
	{
		for (int x = 0; x < image.cols; ++x)
		{
			red.at<Vec3b>(y, x)[0] = 0;
			red.at<Vec3b>(y, x)[1] = 0;

			green.at<Vec3b>(y, x)[0] = 0;
			green.at<Vec3b>(y, x)[2] = 0;

			blue.at<Vec3b>(y, x)[1] = 0;
			blue.at<Vec3b>(y, x)[2] = 0;
		}
	}

	imshow("red", red);
	imshow("green", green);
	imshow("blue", blue);

	waitKey(0);

	return 0;
#endif

#if EX6
	Mat image = imread("../data/lena.png");
	//Mat gray;
	//cvtColor(image, gray, CV_BGR2GRAY);
	//gray.at<unsigned char>(y,x)
	Mat sBGR;
	image.copyTo(sBGR);

	for (int y = 0; y < image.rows; ++y)
	{
		for (int x = 0; x < image.cols; ++x)
		{
			for (int c = 0; c < 3; ++c)
				sBGR.at<Vec3b>(y, x)[c] = (unsigned char)(255 * pow(image.at<Vec3b>(y, x)[c] / 255.0, 2.2) + 0.5);
		}
	}
	
	imshow("lena", image);
	imshow("sRGB", sBGR);

	waitKey(0);

	Mat XYZ;
	cvtColor(sBGR, XYZ, CV_BGR2XYZ);
	imshow("XYZ", XYZ);

	waitKey(0);

	//Mat HSV;
	//cvtColor(sBGR, HSV, CV_BGR2HSV);
	//imshow("HSV", HSV);

	//Mat Lab;
	//cvtColor(sBGR, Lab, CV_BGR2Lab);
	//imshow("Lab", Lab);

	//waitKey(0);

	return 0;
#endif
}