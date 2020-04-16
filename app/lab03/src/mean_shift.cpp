#include <opencv.hpp>   

using namespace std;
using namespace cv;

void main()
{ 
	 //image load
	 Mat img = imread("../data/fruits.png");
	 int width = img.cols;
	 int height = img.rows;

	 Mat outImg;

	 pyrMeanShiftFiltering(img, outImg, 30, 30, 3, TermCriteria(CV_TERMCRIT_ITER+CV_TERMCRIT_EPS, 100, 1.0));

	 //show image
	 imshow("Input", img);
	 imshow("MeanShift filtered", outImg);
	 imwrite("../data/meanshift_filtered.png", outImg);
	 waitKey();

	 // count the number of colors after filtering
	 vector<Vec3b> centers;
	 for (int i = 0; i < height; ++i)
	 {
		 for (int j = 0; j < width; ++j)
		 {
			 if (centers.size() == 0)
			 {
				 centers.push_back(outImg.at<Vec3b>(i, j));
				 continue;
			 }
			 int k = 0;
			 for (;k < centers.size(); ++k)
			 {
				 if (centers[k][0] == outImg.at<Vec3b>(i, j)[0] && centers[k][1] == outImg.at<Vec3b>(i, j)[1] && centers[k][2] == outImg.at<Vec3b>(i, j)[2])
					 break;
			 }
			 if (k == centers.size())
				 centers.push_back(outImg.at<Vec3b>(i, j));
		 }
	 }

	 int K = centers.size();
	 printf("%d", K);
	 for (int k = 0; k < K; ++k)
	 {
		 for (int l = k + 1; l < K; ++l)
		 {
			 if (centers[k][0] == centers[l][0] && centers[k][1] == centers[l][1] && centers[k][2] == centers[l][2])
				 printf("!\n");
		 }
	 }
	 //for (int k = 0; k < K; ++k)
	 //	 printf("%d: %d %d %d\n", k, centers[k][0], centers[k][1], centers[k][2]);
	 Mat random_rgb = Mat::zeros(K, 1, CV_8UC3);
	 for (int i = 0; i < K; ++i)
	 {
		 random_rgb.at<Vec3b>(i)[0] = rand() % 256;
		 random_rgb.at<Vec3b>(i)[1] = rand() % 256;
		 random_rgb.at<Vec3b>(i)[2] = rand() % 256;
	 }


	 Mat res = Mat::zeros(height, width, CV_8UC3);
	 for (int i = 0; i < height; ++i)
	 {
		 for (int j = 0; j < width; ++j)
		 {
			 int k = 0;
			 for (; k < K; ++k)
			 {
				 if (centers[k][0] == outImg.at<Vec3b>(i, j)[0] && centers[k][1] == outImg.at<Vec3b>(i, j)[1] && centers[k][2] == outImg.at<Vec3b>(i, j)[2])
					 break;
			 }
			 res.at<Vec3b>(i, j) = random_rgb.at<Vec3b>(k);
		 }
	 }

	 imshow("MeanShift clusters", res);
	 imwrite("../data/meanshift_random_colored.png", res);
	 waitKey();
}