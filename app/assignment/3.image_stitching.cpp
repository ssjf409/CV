#include <vector>

#include "opencv2/opencv.hpp"  
#include "opencv2/features2d.hpp"
#include "opencv2/xfeatures2d.hpp"

using namespace std;
using namespace cv;
using namespace cv::xfeatures2d;

// Mat get_matches(Mat& img1, Mat& img2): your function for attaining a homography from img1 to img2
// input: gray images img1 and img2
// output: homography from img1 to img2
// This function returns the homography in Mat(3,3,CV_64FC1) format.
cv::Mat get_homography(Mat& img1, Mat& img2)
{
	// a parameter for SURF
	int minHessian = 100;
	Ptr<SURF> surf = SURF::create(minHessian);

	// keypoints from img1 and img2
	vector<KeyPoint> keypoints1, keypoints2;

	// detect keypoints from img1 and img2
	surf->detect(img1, keypoints1);
	surf->detect(img2, keypoints2);

	// describe keypoints from img1 and img2
	Mat descriptors1, descriptors2;
	surf->compute(img1, keypoints1, descriptors1);
	surf->compute(img2, keypoints2, descriptors2);

	// match the keypoints
	FlannBasedMatcher matcher;
	vector< DMatch > dmatches;
	matcher.match(descriptors1, descriptors2, dmatches);

	// store the matching points to data structure used by findHomography
	vector<Point2f> src_pts;
	vector<Point2f> dst_pts;
	for (int i = 0; i < dmatches.size(); i++)
	{
		src_pts.push_back(keypoints1[dmatches[i].queryIdx].pt);
		dst_pts.push_back(keypoints2[dmatches[i].trainIdx].pt);
	}

	// find the homography from src_pts to dst_pts
	cv::Mat H = findHomography(src_pts, dst_pts, CV_RANSAC);

	return H;
}

cv::Mat get_stitched_image(Mat* img, const int nimages)
{
	// gray images
	Mat* img_gray = new Mat[nimages];
	// H[k]: homography from image k-1 to k 
	Mat* H = new Mat[nimages];
	// compositeH[k]: homography from image 0 to k
	Mat* compositeH = new Mat[nimages];
	// icompositeH[k]: homography from image k to 0
	Mat* icompositeH = new Mat[nimages];

	// convert the color images to their gray images
	for (int k = 0; k < nimages; ++k)
	{
		cvtColor(img[k], img_gray[k], COLOR_BGR2GRAY);
	}

	H[0] = Mat::eye(3, 3, CV_64FC1);
	// compute homographies H[k] from image k-1 to k


/*
	
	for (int k = 1; k < nimages; k++)
	{
		int k1 = k - 1;


		H[k] = get_homography(img_gray[k1], img_gray[k]);

	}

	


	// compute composite homographies compositeH[k] from 0 to k
	// compute icomposite homographies icompositeH[k] from k to 0
	compositeH[0] = Mat::eye(3, 3, CV_64FC1);
	icompositeH[0] = Mat::eye(3, 3, CV_64FC1);
	for (int k = 1; k < nimages; ++k)
	{
		int k1 = k - 1;
		compositeH[k] = H[k] * compositeH[k1];
		cout << compositeH[1].at<float>(0, 0) << " " << compositeH[1].at<float>(0, 1) << " " << compositeH[1].at<float>(0, 2) << endl;
		cout << compositeH[1].at<float>(1, 0) << " " << compositeH[1].at<float>(1, 1) << " " << compositeH[1].at<float>(1, 2) << endl;
		cout << compositeH[1].at<float>(2, 0) << " " << compositeH[1].at<float>(2, 1) << " " << compositeH[1].at<float>(2, 2) << endl; // 행렬 모양으로 표시
		printf("행렬 모양으로 표시");
		invert(compositeH[k], icompositeH[k], CV_SVD); //역행렬 만들어 낸다.
		printf("역행렬 계산");
	}	

	// compute the size of the panoramic image
	float min_sx = 0;
	float min_sy = 0;
	float max_ex = (float)(img_gray[0].cols); // 0 -> (int)nimages/2 (중앙 이미지)
	float max_ey = (float)(img_gray[0].rows); //

	for (int k = 1; k < nimages; ++k)
	{
		int sx = 0;
		int ex = img_gray[k].cols;
		int sy = 0;
		int ey = img_gray[k].rows;

		std::vector<Point2f> corners(4);
		corners[0] = cvPoint(0, 0); corners[1] = cvPoint(ex, 0);
		corners[2] = cvPoint(ex, ey); corners[3] = cvPoint(0, ey);

		std::vector<Point2f> tr_corners(4);
		cout << corners[0].x << " " << corners[0].y << endl;
		cout << corners[1].x << " " << corners[1].y << endl;
		cout << corners[2].x << " " << corners[2].y << endl;
		cout << corners[3].x << " " << corners[3].y << endl; // 옮겨질 좌표의 
		printf("옮겨질 영상의 코너 좌표들");


		perspectiveTransform(corners, tr_corners, icompositeH[k]); // 영상 좌표를 이동시키는
		printf("옮겨질 영상의 코너 좌표들을 역행렬 계산해서, 옮겨졌을 때 코너의 좌표들 구함");

		for (int m = 0; m < 4; ++m)
		{
			cout << tr_corners[m].x << " " << tr_corners[m].y << endl;
			printf("옮겨진 코너 좌표들");

			if (tr_corners[m].x < min_sx)
				min_sx = tr_corners[m].x;
			if (tr_corners[m].x > max_ex)
				max_ex = tr_corners[m].x;
			if (tr_corners[m].y < min_sy)
				min_sy = tr_corners[m].y;
			if (tr_corners[m].y > max_ey)
				max_ey = tr_corners[m].y;
		}
	}

	// the panoramic image's coordinate frame is dx and dy shifted from the image frame of img[0]
	// x'=x+dx, y'=y+dy
	int dx = (int)(-min_sx + 0.5); // 영상의 예상 최대 크기가현재 기준 음수 일 수도 있다.
	int dy = (int)(-min_sy + 0.5);

	int width = (int)(max_ex + 0.5) + dx; // 0점으로 땡겨
	int height = (int)(max_ey + 0.5) + dy;

	cout << width<< "    " << height << endl;
	
	// panoramic image
	Mat panorama = Mat::zeros(height, width, CV_8UC3);
	// if any value has been copied to a pixel in the panoramic image the flag is 1, otherwise the flag is 0.
	Mat flag = Mat::zeros(height, width, CV_32SC1); // 1일때는 그리지 않고, 0일때는 그린다.. 이미그려져 있는지 그렇지 않은지
	// copy img[0] to the panoramic image
	for (int y = 0; y < img[0].rows; ++y)
	{
		for (int x = 0; x < img[0].cols; ++x)
		{
			panorama.at<Vec3b>(y + dy, x + dx) = img[0].at<Vec3b>(y, x); // 기준 영상을 +dx +dy 조금 땡겨서 다시 자리 잡는다.
			flag.at<int>(y + dy, x + dx) = 1;
		}
	}

	// the coordinates of the panoramic image is transformed to img[0] first.
	std::vector<Point2f> corners;  // 파노라마 이미지에있는 모든 영상에 대해서 0번재 영상으로 
	for (int y = 0; y < height; ++y)
	{
		for (int x = 0; x < width; ++x)
		{
			corners.push_back(Point2f((float)(x - dx), (float)(y - dy)));
		}
	} //전체 합쳐질 크기의 캔버스에서 기준 영상이 dx,dy 지점부터 차례로 있다. 그 영상을 저장

	int x0, y0, x1, y1;
	// transform a point in the panoramic image to img[k], and copy the corresponding pixels color to the point by using bilinear interpolation.
	for (int k = 1; k<nimages; ++k)
	{
		int rows = img[k].rows;
		int cols = img[k].cols;
		std::vector<Point2f> tr_corners;
		perspectiveTransform(corners, tr_corners, compositeH[k]); //기준영상을 그다음 붙을 영상의 모양으로 변형(형태만)
		cout << tr_corners.size() << endl;

		int m = 0;
		for (int y = 0; y < height; ++y)
		{
			for (int x = 0; x < width; ++x, ++m)
			{				
				if (flag.at<int>(y, x) == 1)
					continue;

				float fx = tr_corners[m].x; //소수점 위치로 떨어지게 된다.
				float fy = tr_corners[m].y;
				x0 = (int)(floor(fx));			//작은 픽셀의 가장 왼쪽 아래와 가장 오른쪽 위의 좌표
				y0 = (int)(floor(fy));
				x1 = (int)(ceil(fx));
				y1 = (int)(ceil(fy));

				if (x0 >= 0 && x1 < cols && y0 >= 0 && y1 < rows) // 옮길 영상의 범위 안에 있으면
				{
					float cx0 = x1 - fx;
					float cx1 = fx - x0;
					float cy0 = y1 - fy;
					float cy1 = fy - y0; // 바이리니어 인터폴레이션

					uchar r = uchar(max(min(cx0*(cy0*img[k].at<Vec3b>(y0, x0)[2] + cy1*img[k].at<Vec3b>(y1, x0)[2]) + cx1*(cy0*img[k].at<Vec3b>(y0, x1)[2] + cy1*img[k].at<Vec3b>(y1, x1)[2]), 255.0f), 0.0f) + 0.5f);
					uchar g = uchar(max(min(cx0*(cy0*img[k].at<Vec3b>(y0, x0)[1] + cy1*img[k].at<Vec3b>(y1, x0)[1]) + cx1*(cy0*img[k].at<Vec3b>(y0, x1)[1] + cy1*img[k].at<Vec3b>(y1, x1)[1]), 255.0f), 0.0f) + 0.5f);
					uchar b = uchar(max(min(cx0*(cy0*img[k].at<Vec3b>(y0, x0)[0] + cy1*img[k].at<Vec3b>(y1, x0)[0]) + cx1*(cy0*img[k].at<Vec3b>(y0, x1)[0] + cy1*img[k].at<Vec3b>(y1, x1)[0]), 255.0f), 0.0f) + 0.5f);

					// 픽셀 소수점 안 맞는 거 분산..

					panorama.at<Vec3b>(y, x) = Vec3b(b, g, r);
					flag.at<int>(y, x) = 1;
				}
			}
		}
	}
*/


	
	for (int k = 1; k < nimages; k++)
	{
		int k1 = k - 1;


		H[k] = get_homography(img_gray[k1], img_gray[k]);

	}


	int mid = int(nimages - 1) / 2; //요거

	printf("%d\n", mid);


	// compute composite homographies compositeH[k] from 0 to k
	// compute icomposite homographies icompositeH[k] from k to 0
	compositeH[0] = Mat::eye(3, 3, CV_64FC1);
	icompositeH[0] = Mat::eye(3, 3, CV_64FC1);
	for (int k = 1; k < nimages; ++k)
	{
		int k1 = k - 1;
		compositeH[k] = H[k] * compositeH[k1];
		cout << compositeH[1].at<float>(0, 0) << " " << compositeH[1].at<float>(0, 1) << " " << compositeH[1].at<float>(0, 2) << endl;
		cout << compositeH[1].at<float>(1, 0) << " " << compositeH[1].at<float>(1, 1) << " " << compositeH[1].at<float>(1, 2) << endl;
		cout << compositeH[1].at<float>(2, 0) << " " << compositeH[1].at<float>(2, 1) << " " << compositeH[1].at<float>(2, 2) << endl; // 행렬 모양으로 표시
		printf("행렬 모양으로 표시");
		invert(compositeH[k], icompositeH[k], CV_SVD); //역행렬 만들어 낸다.
		printf("역행렬 계산");
	}	

	// compute the size of the panoramic image
	float min_sx = 0;
	float min_sy = 0;
	float max_ex = (float)(img_gray[mid].cols); // 0 -> (int)nimages/2 (중앙 이미지)
	float max_ey = (float)(img_gray[mid].rows); //

	for (int k = 1; k < nimages; ++k)
	{
		if (k % 2 == 1) {
			int kodd = mid + (k+1) / 2;

			int sx = 0;
			int ex = img_gray[kodd].cols;
			int sy = 0;
			int ey = img_gray[kodd].rows;

			std::vector<Point2f> corners(4);
			corners[0] = cvPoint(0, 0); corners[1] = cvPoint(ex, 0);
			corners[2] = cvPoint(ex, ey); corners[3] = cvPoint(0, ey);

			std::vector<Point2f> tr_corners(4);
			cout << corners[0].x << " " << corners[0].y << endl;
			cout << corners[1].x << " " << corners[1].y << endl;
			cout << corners[2].x << " " << corners[2].y << endl;
			cout << corners[3].x << " " << corners[3].y << endl; // 옮겨질 좌표의 
			printf("옮겨질 영상의 코너 좌표들\n");


			perspectiveTransform(corners, tr_corners, icompositeH[kodd]*compositeH[mid]); // 영상 좌표를 이동시키는
			//printf("옮겨질 영상의 코너 좌표들을 역행렬 계산해서, 옮겨졌을 때 코너의 좌표들 구함\n");



			for (int m = 0; m < 4; ++m)
			{
				cout << tr_corners[m].x << " " << tr_corners[m].y << endl;
				printf("옮겨진 코너 좌표들\n");

				if (tr_corners[m].x < min_sx)
					min_sx = tr_corners[m].x;
				if (tr_corners[m].x > max_ex)
					max_ex = tr_corners[m].x;
				if (tr_corners[m].y < min_sy)
					min_sy = tr_corners[m].y;
				if (tr_corners[m].y > max_ey)
					max_ey = tr_corners[m].y;
			}


		}
		else {
			int keven = mid - k / 2;

			int sx = 0;
			int ex = img_gray[keven].cols;
			int sy = 0;
			int ey = img_gray[keven].rows;

			std::vector<Point2f> corners(4);
			corners[0] = cvPoint(0, 0); corners[1] = cvPoint(ex, 0);
			corners[2] = cvPoint(ex, ey); corners[3] = cvPoint(0, ey);

			std::vector<Point2f> tr_corners(4);
			cout << corners[0].x << " " << corners[0].y << endl;
			cout << corners[1].x << " " << corners[1].y << endl;
			cout << corners[2].x << " " << corners[2].y << endl;
			cout << corners[3].x << " " << corners[3].y << endl; // 옮겨질 좌표의 
			printf("옮겨질 영상의 코너 좌표들\n");


			perspectiveTransform(corners, tr_corners, compositeH[mid]*icompositeH[keven]); // 영상 좌표를 이동시키는
			//printf("옮겨질 영상의 코너 좌표들을 행렬 계산해서, 옮겨졌을 때 코너의 좌표들 구함\n");

			for (int m = 0; m < 4; ++m)
			{
				cout << tr_corners[m].x << " " << tr_corners[m].y << endl;
				printf("옮겨진 코너 좌표들\n");

				if (tr_corners[m].x < min_sx)
					min_sx = tr_corners[m].x;
				if (tr_corners[m].x > max_ex)
					max_ex = tr_corners[m].x;
				if (tr_corners[m].y < min_sy)
					min_sy = tr_corners[m].y;
				if (tr_corners[m].y > max_ey)
					max_ey = tr_corners[m].y;
			}
		}
		printf("%d번째\n", k);
		printf("%f : min_sx\n", min_sx);
		printf("%f : max_ex\n", max_ex);
		printf("%f : min_sy\n", min_sy);
		printf("%f : min_ey\n", max_ey);

	}

	// the panoramic image's coordinate frame is dx and dy shifted from the image frame of img[0]
	// x'=x+dx, y'=y+dy
	int dx = (int)(-min_sx + 0.5); // 영상의 예상 최대 크기가현재 기준 음수 일 수도 있다.
	int dy = (int)(-min_sy + 0.5);

	int width = (int)(max_ex + 0.5) + dx; // 0점으로 땡겨
	int height = (int)(max_ey + 0.5) + dy;

	cout << width<< "    " << height << endl;
	
	// panoramic image
	Mat panorama = Mat::zeros(height, width, CV_8UC3);
	// if any value has been copied to a pixel in the panoramic image the flag is 1, otherwise the flag is 0.
	Mat flag = Mat::zeros(height, width, CV_32SC1); // 1일때는 그리지 않고, 0일때는 그린다.. 이미그려져 있는지 그렇지 않은지
	// copy img[0] to the panoramic image
	for (int y = 0; y < img[mid].rows; ++y)
	{
		for (int x = 0; x < img[mid].cols; ++x)
		{
			panorama.at<Vec3b>(y + dy, x + dx) = img[mid].at<Vec3b>(y, x); // 기준 영상을 +dx +dy 조금 땡겨서 다시 자리 잡는다.
			flag.at<int>(y + dy, x + dx) = 1;
		}
	}

	// the coordinates of the panoramic image is transformed to img[0] first.
	std::vector<Point2f> corners;  // 파노라마 이미지에있는 모든 영상에 대해서 0번재 영상으로 
	for (int y = 0; y < height; ++y)
	{
		for (int x = 0; x < width; ++x)
		{
			corners.push_back(Point2f((float)(x - dx), (float)(y - dy)));
		}
	} //전체 합쳐질 크기의 캔버스에서 기준 영상이 dx,dy 지점부터 차례로 있다. 그 영상을 저장

	int x0, y0, x1, y1;
	// transform a point in the panoramic image to img[k], and copy the corresponding pixels color to the point by using bilinear interpolation.
	for (int k = 1; k<nimages; ++k)
	{
		if (k % 2 == 1) {
			int kodd = mid + (k + 1) / 2;

			int rows = img[kodd].rows;
			int cols = img[kodd].cols;
			std::vector<Point2f> tr_corners;
			perspectiveTransform(corners, tr_corners, compositeH[kodd]*icompositeH[mid]*compositeH[0]); //기준영상을 그다음 붙을 영상의 모양으로 변형(형태만) //compositeH[kodd]
			cout << tr_corners.size() << endl;

			int m = 0;
			for (int y = 0; y < height; ++y)
			{
				for (int x = 0; x < width; ++x, ++m)
				{
					if (flag.at<int>(y, x) == 1)
						continue;

					float fx = tr_corners[m].x; //소수점 위치로 떨어지게 된다.
					float fy = tr_corners[m].y;
					x0 = (int)(floor(fx));			//작은 픽셀의 가장 왼쪽 아래와 가장 오른쪽 위의 좌표
					y0 = (int)(floor(fy));
					x1 = (int)(ceil(fx));
					y1 = (int)(ceil(fy));

					if (x0 >= 0 && x1 < cols && y0 >= 0 && y1 < rows) // 옮길 영상의 범위 안에 있으면
					{
						float cx0 = x1 - fx;
						float cx1 = fx - x0;
						float cy0 = y1 - fy;
						float cy1 = fy - y0; // 바이리니어 인터폴레이션

						uchar r = uchar(max(min(cx0*(cy0*img[kodd].at<Vec3b>(y0, x0)[2] + cy1 * img[kodd].at<Vec3b>(y1, x0)[2]) + cx1 * (cy0*img[kodd].at<Vec3b>(y0, x1)[2] + cy1 * img[kodd].at<Vec3b>(y1, x1)[2]), 255.0f), 0.0f) + 0.5f);
						uchar g = uchar(max(min(cx0*(cy0*img[kodd].at<Vec3b>(y0, x0)[1] + cy1 * img[kodd].at<Vec3b>(y1, x0)[1]) + cx1 * (cy0*img[kodd].at<Vec3b>(y0, x1)[1] + cy1 * img[kodd].at<Vec3b>(y1, x1)[1]), 255.0f), 0.0f) + 0.5f);
						uchar b = uchar(max(min(cx0*(cy0*img[kodd].at<Vec3b>(y0, x0)[0] + cy1 * img[kodd].at<Vec3b>(y1, x0)[0]) + cx1 * (cy0*img[kodd].at<Vec3b>(y0, x1)[0] + cy1 * img[kodd].at<Vec3b>(y1, x1)[0]), 255.0f), 0.0f) + 0.5f);

						// 픽셀 소수점 안 맞는 거 분산..

						panorama.at<Vec3b>(y, x) = Vec3b(b, g, r);
						flag.at<int>(y, x) = 1;
					}
				}
			}
		}
		else {
			int keven = mid - k / 2;

			int rows = img[keven].rows;
			int cols = img[keven].cols;
			std::vector<Point2f> tr_corners;
			perspectiveTransform(corners, tr_corners, icompositeH[mid]*compositeH[keven]); //기준영상을 그다음 붙을 영상의 모양으로 변형(형태만)
			cout << tr_corners.size() << endl;

			int m = 0;
			for (int y = 0; y < height; ++y)
			{
				for (int x = 0; x < width; ++x, ++m)
				{
					if (flag.at<int>(y, x) == 1)
						continue;

					float fx = tr_corners[m].x; //소수점 위치로 떨어지게 된다.
					float fy = tr_corners[m].y;
					x0 = (int)(floor(fx));			//작은 픽셀의 가장 왼쪽 아래와 가장 오른쪽 위의 좌표
					y0 = (int)(floor(fy));
					x1 = (int)(ceil(fx));
					y1 = (int)(ceil(fy));

					if (x0 >= 0 && x1 < cols && y0 >= 0 && y1 < rows) // 옮길 영상의 범위 안에 있으면
					{
						float cx0 = x1 - fx;
						float cx1 = fx - x0;
						float cy0 = y1 - fy;
						float cy1 = fy - y0; // 바이리니어 인터폴레이션

						uchar r = uchar(max(min(cx0*(cy0*img[keven].at<Vec3b>(y0, x0)[2] + cy1 * img[keven].at<Vec3b>(y1, x0)[2]) + cx1 * (cy0*img[keven].at<Vec3b>(y0, x1)[2] + cy1 * img[keven].at<Vec3b>(y1, x1)[2]), 255.0f), 0.0f) + 0.5f);
						uchar g = uchar(max(min(cx0*(cy0*img[keven].at<Vec3b>(y0, x0)[1] + cy1 * img[keven].at<Vec3b>(y1, x0)[1]) + cx1 * (cy0*img[keven].at<Vec3b>(y0, x1)[1] + cy1 * img[keven].at<Vec3b>(y1, x1)[1]), 255.0f), 0.0f) + 0.5f);
						uchar b = uchar(max(min(cx0*(cy0*img[keven].at<Vec3b>(y0, x0)[0] + cy1 * img[keven].at<Vec3b>(y1, x0)[0]) + cx1 * (cy0*img[keven].at<Vec3b>(y0, x1)[0] + cy1 * img[keven].at<Vec3b>(y1, x1)[0]), 255.0f), 0.0f) + 0.5f);

						// 픽셀 소수점 안 맞는 거 분산..

						panorama.at<Vec3b>(y, x) = Vec3b(b, g, r);
						flag.at<int>(y, x) = 1;
					}
				}
			}

		}
	}


	delete[] img_gray;
	delete[] H;
	delete[] compositeH;
	delete[] icompositeH;

	return panorama;
}

int main()
{
	// number of images to be loaded
	// you can decrease the number of images if you cannot stitch all 9 images together
	const int nimages = 9; // 영상 3개를 이어붙이고 싶구나
	// color images
	Mat img[nimages];

	// file name
	char fn[1024];
	// center image index

	// load images and convert the color images to their gray images
	for (int k = 0; k < nimages; ++k)
	{
		sprintf(fn, "../data/songdo%02d.jpg", k);
		img[k] = imread(fn);
	}

	//inu_panorama ip;
	Mat flag;
	Mat panorama = get_stitched_image(img, nimages);

	imwrite("../result/result.jpg", panorama);
	imshow("panorama", panorama);
	Mat resized_panorama;
	resize(panorama, resized_panorama, Size(1024, 512), 0, 0, CV_INTER_LINEAR);
	imshow("resized_panorama", resized_panorama);
	waitKey(0);
	return 0;
}