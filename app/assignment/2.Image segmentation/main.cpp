#include <opencv.hpp>

// Include necessary files here.

#include "SLIC.h"

#define K 200
#define compactness 5

using namespace std;
using namespace cv;


int main()
{
	// Load the image.
	Mat img = imread("../data/beach.jpg");

	// Show the image.
	imshow("image", img);
	waitKey(0);

	// Write your code for calling others' functions for image segmentatio

	int width = img.cols;
	int height = img.rows;
	int imgSize = width * height;
	int dim = img.channels();


	unsigned int *image;

	int* label = new int[imgSize];
	int numlabels(0);



	SLIC slic;

	unsigned char *pImage = new unsigned char[imgSize * 4];



	for (int j = 0; j < height; j++) {
		Vec3b * ptr = img.ptr<Vec3b>(j);
		for (int i = 0; i < width; i++) {
			pImage[j * width * 4 + 4 * i + 3] = 0;
			pImage[j * width * 4 + 4 * i + 2] = ptr[0][2];//R
			pImage[j * width * 4 + 4 * i + 1] = ptr[0][1];//G
			pImage[j * width * 4 + 4 * i] = ptr[0][0];//B		
			ptr++;
		}
	}


	image = new unsigned int[imgSize];

	memcpy(image, (unsigned int*)pImage, imgSize * sizeof(unsigned int));
	delete pImage;


	slic.DoSuperpixelSegmentation_ForGivenNumberOfSuperpixels(image, width, height, label, numlabels,
		K, compactness);

	slic.DrawContoursAroundSegments(image, label, width, height, 0);

	


	 

	// Save the segmentation result to Mat result.


	// I temporarily copy the original image to the result.
	Mat result = img.clone();

	result.create(height, width, CV_8UC3);
	for (int j = 0; j < height; ++j)
	{

		Vec3b * p = result.ptr<Vec3b>(j);
		for (int i = 0; i < width; ++i)
		{
			p[i][0] = (unsigned char)(image[j*width + i] & 0xFF); //Blue 
			p[i][1] = (unsigned char)((image[j*width + i] >> 8) & 0xFF); //Green 
			p[i][2] = (unsigned char)((image[j*width + i] >> 16) & 0xFF); //Red 
		}
	}



	imshow("result", result);
	waitKey(0);

	return 0;
}