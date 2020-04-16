#include <string>
#include <vector>
#include <stdio.h>

#include "opencv2/opencv.hpp"

#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <vector>


#define max_r 110
#define min_r 50

using namespace std;
using namespace cv;

void detectHScolor(const Mat& image, double minHue, double maxHue, double maxSat, Mat& mask) {

	Mat hsv;
	cvtColor(image, hsv, CV_BGR2HSV);

	vector<Mat> channels;
	split(hsv, channels);

	Mat mask1;
	threshold(channels[0], mask1, maxHue, 255, THRESH_BINARY_INV);
	Mat mask2;
	threshold(channels[0], mask2, minHue, 255, THRESH_BINARY);
	Mat hueMask;
	if (minHue < maxHue) hueMask = mask1 & mask2;
	else hueMask = mask1 | mask2;

	threshold(channels[1], mask1, maxSat, 255, THRESH_BINARY_INV);
	threshold(channels[1], mask1, maxSat, 255, THRESH_BINARY);

	Mat satMask;
	satMask = mask1 & mask2;

	mask = hueMask & satMask;
}




int main(int argc, char** argv)
{
	// Write your path of the training images.
	String path("../data/training/*.jpg");
	vector<String> fn;
	glob(path, fn, false);

	int num_images = (int)(fn.size());
	for (int k = 0; k<num_images; ++k)
	{
		// Load image(write number of image)
		//printf("%s\n", fn[k].c_str());
		Mat src = imread(fn[k]);

		// Extract the file name without its extension.
		string str(fn[k].c_str());
		size_t pos1 = str.find_last_of('/');
		size_t pos2 = str.find_last_of('\\');
		size_t pos3 = str.find_last_of('.');
		size_t pos = pos1;
		if (pos2 != string::npos)
			pos = pos2;

		string substr = str.substr(pos + 1, pos3 - pos - 1);
		printf("%s\n", substr.c_str());

		if (src.empty())
		{
			printf("Please check the data path.\n");
			return 0;
		}

		// Variable for saving circles
		// cirlces[m][0] : x coordinate of the center of the m'th circle
		// circles[m][1] : y coordinate of the center of the m'th circle
		// circles[m][2] : radius of the m'th circle
		// The circles should be in order of their "circleness".
		// We will check only the first circle, i.e., circles[0].
		vector<Vec3f> circles;

		// Write your circle detect code here. 
		// We are going to copy and paste your code here.
		// Below is an example of using the "HoughCircles" function in OpenCV.
		// If you just use or optimize this function, your score will be multiplied by 0.5 






		int intensity_b = src.at<Vec3b>(234, 145)[0];			//B

		int intensity_g = src.at<Vec3b>(234, 145)[1];			//G

		int intensity_r = src.at<Vec3b>(234, 145)[2];				//R

		//printf("%d %d %d \n", intensity_b, intensity_g, intensity_r);

		Mat src_hsv;
		cvtColor(src, src_hsv, CV_RGB2HSV);


		int intensity_h = src_hsv.at<Vec3b>(234, 145)[0];			//H

		int intensity_s = src_hsv.at<Vec3b>(234, 145)[1];			//S

		int intensity_v = src_hsv.at<Vec3b>(234, 145)[2];			//V

		//printf("%d %d %d \n", intensity_h, intensity_s, intensity_v);



		Mat mask;
		detectHScolor(src, 124, 207, 69, mask); // 15다음에 , 255은 왜 빠지지?

		Mat detected(src.size(), CV_8UC3, Scalar(0, 0, 0));
		src.copyTo(detected, mask);

		//namedWindow("Mask");
		//imshow("Mask", mask);
		//namedWindow("Detected");
		//imshow("Detected", detected);


		//waitKey();

		GaussianBlur(mask, mask, Size(5, 5), 2, 2);

		//namedWindow("Mask_gaussian");
		//imshow("Mask_gaussian", mask);
		//waitKey();


		int width = src.cols - 1;
		int height = src.rows - 1;

		int sum_x = 0;
		int sum_y = 0;
		int cnt = 0;


		for (int j = 0; j < height; j++) {
			for (int i = 0; i < width; i++) {
				if (mask.at<uchar>(j, i) == 255) {
					cnt++;
					sum_x += i;
					sum_y += j;

				}
			}
		}
		int cx = 0;
		int cy = 0;

		if (cnt != 0) {
			cx = round(sum_x / cnt);
			cy = round(sum_y / cnt);
		}
		circle(src, { cx, cy }, 1, Scalar(255, 0, 255), 3, LINE_AA);

		//imshow("test", src);
		//waitKey();

		int up = cy;
		int down = height - cy;
		int right = cx;
		int left = width - cx;

		int candi_r_up = 0;
		int candi_r_down = 0;
		int candi_r_left = 0;
		int candi_r_right = 0;




		for (int j = cy; j > 0; j--) {
			if ((mask.at<uchar>(j, cx) != 0) && (cy - j > candi_r_up)) {
				candi_r_up = cy - j;
			}
		}
		for (int j = cy; j < height; j++) {
			if ((mask.at<uchar>(j, cx) != 0) && (j - cy > candi_r_down)) {
				candi_r_down = j - cy;
			}
		}
		for (int i = cx; i > 0; i--) {
			if ((mask.at<uchar>(cy, i) != 0) && (cx - i > candi_r_left)) {
				candi_r_left = cx - i;
			}
		}
		for (int i = cx; i < width; i++) {
			if ((mask.at<uchar>(cy, i) != 0) && (i - cx > candi_r_right)) {
				candi_r_right = i - cx;
			}
		}
		//printf("%d\n", candi_r_up);
		//printf("%d\n", candi_r_down);
		//printf("%d\n", candi_r_left);
		//printf("%d\n", candi_r_right);


		int aver_r = 0;
		int aver_cnt = 0;

		if (candi_r_up < max_r && candi_r_up > min_r) {
			aver_r += candi_r_up;
			aver_cnt++;
		}
		if (candi_r_down < max_r && candi_r_down > min_r) {
			aver_r += candi_r_down;
			aver_cnt++;
		}
		if (candi_r_left < max_r && candi_r_left > min_r) {
			aver_r += candi_r_left;
			aver_cnt++;
		}
		if (candi_r_right < max_r && candi_r_right > min_r) {
			aver_r += candi_r_right;
			aver_cnt++;
		}

		int r = 75;
		if (aver_cnt != 0) {
			int r = round(aver_r / aver_cnt);
			//printf("원 반지름 : %d\n", r);
		}




		//circle(src, { cx, cy }, r, Scalar(255, 0, 255), 3, LINE_AA);




		circles.clear();


		circles.push_back({ (float)cx,(float)cy,(float)r });
		//printf("x,y,r : %f %f %f\n", (float)cx, (float)cy, (float)r);




		// Please do not change the below, which saves the circle parameters and result images.

		Vec3i c = circles[0];
		Point center = Point(c[0], c[1]);
		// Circle center
		circle(src, center, 1, Scalar(0, 255, 0), 3, LINE_AA);
		// Circle outline
		int radius = c[2];
		circle(src, center, radius, Scalar(255, 0, 255), 3, LINE_AA);


		// Write the circle parameters. 
		char fo[1024];
		sprintf_s(fo, "../result/%s.txt", substr.c_str());
		FILE* fpo = fopen(fo, "w");
		// Wrtie your center x,y & radius 
		// Write only one coordinates and radius per image.
		fprintf(fpo, "%f %f %f\n", circles[0][0], circles[0][1], circles[0][2]);

		fclose(fpo);

		// Write your result image in the result folder
		imwrite(format("../result/%s.jpg", substr.c_str()), src);


	}
}
