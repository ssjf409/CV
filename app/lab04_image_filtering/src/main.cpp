#include <opencv2/opencv.hpp>

using namespace std;
using namespace cv;

int main()
{
	// Load the input image.
	Mat img = imread("../data/cat.jpg", CV_LOAD_IMAGE_GRAYSCALE);

	// Generate salt and pepper noise.
	Mat saltpepper_noise = Mat::zeros(img.rows, img.cols, CV_8U);
	randu(saltpepper_noise, 0, 255);
	float amount = 0.1;
	float half_amount = 0.5*amount;
	float black_thresh = 255 * half_amount;
	float white_thresh = 255 * (1 - half_amount);

	Mat black = saltpepper_noise < black_thresh;
	Mat white = saltpepper_noise > white_thresh;

	// Add salt and pepper noise.
	Mat noisy = img.clone();
	noisy.setTo(255, white);
	noisy.setTo(0, black);

	imshow("saltpepper", noisy);

	Mat bilateral, gaussian, median;
	bilateralFilter(noisy, bilateral, 9, 75, 75);
	imshow("bilateral", bilateral);
	GaussianBlur(noisy, gaussian, Size(5, 5), 2 / 3.0);
	imshow("Gaussian", gaussian);
	medianBlur(noisy, median, 5);
	imshow("median", median);

	waitKey();
}