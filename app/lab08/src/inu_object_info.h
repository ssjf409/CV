#ifndef inu_object_info_h
#define inu_object_info_h

#include "opencv2/opencv.hpp"  

class inu_object_info
{
public:
	int m_id;			// object id
	cv::Mat m_A;	// the affine transformation from the model to the image
	cv::Point2f m_center;

	inu_object_info() {}
	virtual ~inu_object_info() {}
};

#endif