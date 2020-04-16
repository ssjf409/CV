#ifndef inu_generalized_hough_transform_h
#define inu_generalized_hough_transform_h

#include <vector>
#include "opencv2/opencv.hpp"  

class inu_generalized_hough_transform
{
public:
	// number of bins in each direction
	int m_nlabels;
	int m_nx;
	int m_ny;
	int m_ns;
	int m_no;
	// the spacing between bins
	double m_step_x;
	double m_step_y;
	double m_step_s;
	double m_step_o;
	// hough bin centers: x, y, scale, orientation
	std::vector<int> m_label;
	std::vector<float> m_x;
	std::vector<float> m_y;
	std::vector<float> m_s;
	std::vector<float> m_o;

	// hough bins
	std::vector<std::vector<std::vector<std::vector<std::vector<int>>>>> m_bin;
	// keypoint indices voted for each bin
	std::vector<std::vector<std::vector<std::vector<std::vector<std::vector<cv::Vec2i>>>>>> m_index;

	inu_generalized_hough_transform();
	virtual ~inu_generalized_hough_transform();

	// initialize hough bins
	void init(std::vector<int>& labels, float min_x, float max_x, float min_y, float max_y, float min_scale, float max_scale, float res_x = 2.0f, float res_y = 2.0f, float res_s = 1.5f, float res_ori = 10.0f);
	// vote
	void vote(int l, float x, float y, float s, float o, cv::Vec2i& ind);
	// get object labels and keypoint indexes with sufficient number of inliers (or entries)
	std::vector<int> get_object_labels(std::vector<std::vector<cv::Vec2i>>& indices, int thresh=3);
};

#endif //inu_generalized_hough_transform_h