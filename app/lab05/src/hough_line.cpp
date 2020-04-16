#include "opencv2/opencv.hpp"

using namespace std;
using namespace cv;

int main()
{
	// Declare the output variables
	Mat dst, cdst, cdstP;
	const char* filename = "../data/chess.jpg";
	// Loads an image
	Mat src = imread(filename, IMREAD_GRAYSCALE);
	// Check if image is loaded fine
	
	if (src.empty()) {
		printf(" Error opening image\n");
		return -1;
	}
	// Edge detection
	Canny(src, dst, 50, 200, 3);
	// Copy edges to the images that will display the results in BGR
	cvtColor(dst, cdst, COLOR_GRAY2BGR);
	cdstP = cdst.clone();
	// Standard Hough Line Transform
	vector<Vec2f> lines; // will hold the results of the detection
	HoughLines(dst, lines, 1, CV_PI / 180, 150, 0, 0); // runs the actual detection
													   // Draw the lines
	for (size_t i = 0; i < lines.size(); i++)
	{
		float rho = lines[i][0], theta = lines[i][1];
		Point pt1, pt2;
		double a = cos(theta), b = sin(theta);
		double x0 = a * rho, y0 = b * rho;
		pt1.x = cvRound(x0 + 1000 * (-b));
		pt1.y = cvRound(y0 + 1000 * (a));
		pt2.x = cvRound(x0 - 1000 * (-b));
		pt2.y = cvRound(y0 - 1000 * (a));
		line(cdst, pt1, pt2, Scalar(0, 0, 255), 2, LINE_AA);
	}
		
	imshow("Source", src);
	imshow("Detected Lines (in red) - Standard Hough Line Transform", cdst);
	waitKey();
	
	return 0;
}