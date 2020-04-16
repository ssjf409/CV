#include <opencv.hpp>
#include <iostream>

using namespace std;
using namespace cv;


int main()
{
	// input image: img_input
	// gray image: gray_img
	// result: img_result
	Mat img_input, gray_img, img_result;

	// 0. Read the input image.
	img_input = imread("../data/lena.jpg");
	// 1. Convert the color image to the gray image.
	cvtColor(img_input, gray_img, CV_BGR2GRAY);
	// If the image is not loaded, exit the program.
	if (img_input.empty())
	{
		cout << "파일을 읽어올 수 없습니다. 경로나 파일명을 다시 확인하세요.\n" << endl;
		exit(1);
	}

	// 2. Histogram computation
	// This is the array for the unnormalized histogram.
	vector<float> histogram;
	for (int i = 0; i < 256; ++i)
		histogram.push_back(0.0f);

	// Write your code here for building the (unnormalized) histogram here.

	for (int y = 0; y < gray_img.rows; ++y) {
		for (int x = 0; x < gray_img.cols; ++x) {
			int value = gray_img.at<uchar>(y, x);
			histogram[value]++; // 다음 벡터 어떻게 하나?
		}
	}



	// 3. Histogram normalization
	// This is the array for the normalized histogram. 
	vector<float> normalized_histogram;
	for (int i = 0; i < 256; ++i)
		normalized_histogram.push_back(0.0f);

	// Write your code here for normalizing the histogram.
	for (int i = 0; i < 256; i++) {
		normalized_histogram[i] = histogram[i] / (gray_img.rows * gray_img.cols);
	}
	


	// 4. cdf computation
	// This is the array for the cumulative histogram.
	vector<float> cumulative_histogram;
	for (int i = 0; i < 256; ++i)
		cumulative_histogram.push_back(0.0f);

	// Write your code here for computing the cumulative histogram.
	float sum = 0;
	for (int i = 1; i < 256; i++) {
		sum += normalized_histogram[i];
		cumulative_histogram[i] = sum;
	}


	// 5. computation of the mapping function l_out
	// This is the array for the mapping function (look-up table).
	vector<unsigned char> l_out;
	for (int i = 0; i < 256; ++i)
		l_out.push_back(0);

	// Write your code here for computing the l_out function here

	// 이렇게 l out 만 구하는 코드만 적으면 되는 건지?
	for (int y = 0; y < img_input.rows; y++) {
		for (int x = 0; x < img_input.cols; x++) {
			int value = gray_img.at<uchar>(y, x);
			l_out[value] = round(cumulative_histogram[value] * 255);
		}
	}


	// 6. Histogram normalization
	// This is the Mat in which histogram-equalized image will be stored.
	img_result = Mat::zeros(img_input.rows, img_input.cols, CV_8UC1);
	
	// Write your code here for equalizing the histogram of gray_img.

	//int histogram_equal[256] = { 0, };

	
	for (int y = 0; y < img_input.rows; y++) {
		for (int x = 0; x < img_input.cols; x++) {
			img_result.at<uchar>(y, x) = round(cumulative_histogram[gray_img.at<uchar>(y, x)] * 255);
			//histogram_equal[img_result.at<uchar>(y, x)] += 1;
		}
	}

	
	// 7. Show your results
	imshow("Original (gray)", gray_img);
	imshow("Histogram equalized", img_result);
	
	waitKey();

	return 0;

}