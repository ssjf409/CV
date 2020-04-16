#include <iostream>
#include <vector>
#include <iomanip>
#include "opencv2/highgui.hpp"
#include "opencv2/objdetect/objdetect.hpp"
#include "opencv2/videoio.hpp"
#include "opencv2/imgproc.hpp"
#include "opencv2/text.hpp"
#include "opencv2/text/erfilter.hpp"

using namespace std;
using namespace cv;

int main()
{
	//Read Image & Classifier
	Mat src = imread("../data/cvImage.jpg");

	vector<Mat> channels;
	
	String text_classifier_name1 = "../data/trained_classifierNM1.xml";
	String text_classifier_name2 = "../data/trained_classifierNM2.xml";
	String erGrouping = "../data/trained_classifier_erGrouping.xml";


	//compute channel
	cv::text::computeNMChannels(src, channels);
	for (int i = 0; i < 4; ++i)
	{
		channels.push_back(255 - channels[i]);
	}

	//for (int i = 0; i < channels.size(); i++)
	//{
	//	stringstream ss;
	//	ss << "Channel: " << i;
	//	imshow(ss.str(), channels.at(i));
	//}
	Ptr<text::ERFilter> er_filter1 = text::createERFilterNM1(cv::text::loadClassifierNM1(text_classifier_name1), 16, 0.00015f, 0.13f, 0.2f, true, 0.1f);
	Ptr<text::ERFilter> er_filter2 = text::createERFilterNM2(cv::text::loadClassifierNM2(text_classifier_name2), 0.5);

	vector<vector<cv::text::ERStat> > regions(channels.size());

	for (int i = 0; i < (int)channels.size(); i++)
	{
		er_filter1->run(channels[i], regions[i]);
		er_filter2->run(channels[i], regions[i]);
	}

	for (int i = 0; i < (int)channels.size(); i++)
	{
		Mat dst = Mat::zeros(channels[0].rows + 2, channels[0].cols + 2, CV_8UC1);
		for (int j = 0; j < (int)regions[i].size(); j++)
		{
			cv::text::ERStat er = regions[i][j];
			if (er.parent != NULL)
			{
				int newmaskvalue = 255;
				int flags = 4 + (newmaskvalue << 8) + FLOODFILL_FIXED_RANGE + FLOODFILL_MASK_ONLY;
				floodFill(channels[i], dst, Point(er.pixel % channels[i].cols, er.pixel / channels[i].cols), Scalar(255), 0, Scalar(er.level), Scalar(0), flags);
			}
		}
		//stringstream ss;
		//ss << "Regions/Channel: " << i;
		//imshow(ss.str(), dst);
	}

	vector<Rect> groups;
	vector< vector<Vec2i> > region_groups;
	cv::text::erGrouping(src, channels, regions, region_groups, groups, cv::text::ERGROUPING_ORIENTATION_ANY, erGrouping, 0.5);
	for (int i = (int)groups.size() - 1; i >= 0; i--)
	{
		if (src.type() == CV_8UC3)
			rectangle(src, groups.at(i).tl(), groups.at(i).br(), Scalar(0, 255, 255), 3, 8);
		else
			rectangle(src, groups.at(i).tl(), groups.at(i).br(), Scalar(255, 0, 0), 3, 8);
	}
	imshow("grouping", src);
	cv::waitKey(10000);
	er_filter1.release();
	er_filter2.release();
	regions.clear();
	groups.clear();

}
