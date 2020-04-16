#include <opencv.hpp>

using namespace std;
using namespace cv;

int main()
{
	// k: The number of clusters
	const int K = 4;

	// Load the image
	Mat img = imread("../data/fruits.png");

	// width and height of the image
	int width = img.cols;
	int height = img.rows;
	// the number of pts (pixels)
	int npts = width * height;

	// The points to be clustered, dimension: 3 (bgr)
	Mat pts=Mat::zeros(npts, 1, CV_32FC3);
	// k centers
	Mat centers=Mat::zeros(K, 1, CV_32FC3);
	// labels of pixels
	Mat labels;

	int m = 0;
	for (int i = 0; i < height; ++i)
	{
		for (int j = 0; j < width; ++j, ++m)
		{
			pts.at<Vec3f>(m)[0] = img.at<Vec3b>(i, j)[0];
			pts.at<Vec3f>(m)[1] = img.at<Vec3b>(i, j)[1];
			pts.at<Vec3f>(m)[2] = img.at<Vec3b>(i, j)[2];
			//printf("(%f\t%f\t%f)\n", pts.at<Vec3f>(m)[0], pts.at<Vec3f>(m)[1], pts.at<Vec3f>(m)[2]);
		}
	}

	kmeans(pts, K, labels, TermCriteria(CV_TERMCRIT_ITER + CV_TERMCRIT_EPS, 10, 1.0), 
		3, KMEANS_PP_CENTERS, centers);

	Mat result = Mat::zeros(height, width, CV_8UC3);

	m = 0;
	for (int i = 0; i < height; ++i)
	{
		for (int j = 0; j < width; ++j, ++m)
		{
			int l = labels.at<int>(m);

			result.at<Vec3b>(i, j)[0] = (uchar)round(centers.at<Vec3f>(l)[0]);
			result.at<Vec3b>(i, j)[1] = (uchar)round(centers.at<Vec3f>(l)[1]);
			result.at<Vec3b>(i, j)[2] = (uchar)round(centers.at<Vec3f>(l)[2]);
		}
	}

	imshow("Input", img);
	imshow("Mean-colored image", result);
	//imwrite("../data/k-means_rgb_mean_colored.png", result);
	waitKey(0);

	Mat random_rgb = Mat::zeros(K, 1, CV_8UC3);
	for (int i = 0; i < K; ++i)
	{
		random_rgb.at<Vec3b>(i)[0] = rand() % 256;
		random_rgb.at<Vec3b>(i)[1] = rand() % 256;
		random_rgb.at<Vec3b>(i)[2] = rand() % 256;
	}

	m = 0;
	for (int i = 0; i < height; ++i)
	{
		for (int j = 0; j < width; ++j, ++m)
		{
			int l = labels.at<int>(m);
			//printf("l=%d\n", l);

			result.at<Vec3b>(i, j) = random_rgb.at<Vec3b>(l);
		}
	}
	imshow("Clusters", result);

	waitKey(0);

	return 0;
}