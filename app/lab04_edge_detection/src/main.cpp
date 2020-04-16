#include <opencv2/opencv.hpp>

using namespace std;
using namespace cv;

int main()
{
	// Load the input image.
	Mat img = imread("../data/cat.jpg", CV_LOAD_IMAGE_GRAYSCALE);
	Mat blurred;
	GaussianBlur(img, blurred, Size(3, 3), 1.0 / 3.0);

	namedWindow("Canny", CV_WINDOW_AUTOSIZE);

	int low_value;
	int high_value;
	createTrackbar("low threshold", "Canny", &low_value, 1000);
	createTrackbar("high threshold", "Canny", &high_value, 1000);
	setTrackbarPos("low threshold", "Canny", 50);
	setTrackbarPos("high threshold", "Canny", 100);

	Mat edges;
	while (true) {

		low_value = getTrackbarPos("low threshold", "Canny");
		high_value = getTrackbarPos("high threshold", "Canny");

		Canny(blurred, edges, low_value, high_value);

		imshow("Canny", edges);

		if (waitKey(1) == 'q') break;
	}

	waitKey(0);
	return 0;
}