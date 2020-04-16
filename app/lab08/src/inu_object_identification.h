#ifndef inu_object_identification_h
#define inu_object_identification_h

#include <vector>

#include "opencv2/opencv.hpp"  
#include "opencv2/features2d.hpp"
#include "opencv2/xfeatures2d.hpp"

#include "inu_object_info.h"

class inu_object_identification
{
public:
	// number of models
	int m_nmodels;
	// id of the fist object
	int m_first;
	// tree structure for fast indexing of descriptor vectors
	cv::flann::Index* m_kd_tree;
	// descriptor dimension
	int m_dim;
	// descriptor vectors for training
	cv::Mat m_train_feature;
	// training labels
	std::vector<int> m_train_label;

	// the remaining are used for the generalized hough transform
	// model centers one point per model, the index begins from zero that corresponds m_first
	std::vector<cv::Point2f> m_train_center;
	// feature locations
	std::vector<cv::Point2f> m_train_pt;
	// feature orientations
	std::vector<float> m_train_angle;
	// feature scales
	std::vector<float> m_train_size;

	inu_object_identification();
	virtual ~inu_object_identification();

	// random permutation used in RANSAC
	inline void rand_perm(std::vector<int>& index, std::vector<double>& val, int n, int k)
	{
		index.resize(n);
		val.resize(n);

		int i;
		for (i = 0; i<n; ++i)
		{
			index[i] = i;
			val[i] = rand() / (double)RAND_MAX;
		}

		int j, itemp;
		double temp;

		for (i = 0; i<k; ++i)
		{
			for (j = i + 1; j<n; ++j)
			{
				if (val[i]>val[j])
				{
					temp = val[i];
					val[i] = val[j];
					val[j] = temp;

					itemp = index[i];
					index[i] = index[j];
					index[j] = itemp;
				}
			}
		}
	}

	// build the knn tree from KAIST 104 DB
	int train_knn_tree_from_KAIST_104(const int nmodels = 104, const char* db_path="../data/KAIST-104/DB");
	// find objects from img
	std::vector<inu_object_info> find_objects(cv::Mat& img, int knn=1, int min_num_inliers=4, float thresh_inlier=5.0f);
	// copute affine transform from src_pts to dst_pts
	cv::Mat get_affine_transform_RANSAC(std::vector<cv::Point2f>& src_pts, std::vector<cv::Point2f>& dst_pts, float thresh = 5.0f, int max_iter=100);
	
	// functions for drawing objects
	cv::Mat get_warped_KAIST_104_model_image(inu_object_info& info, cv::Mat& img, const char* db_path = "../data/KAIST-104/DB");
	cv::Mat get_combined_image(cv::Mat& warped, cv::Mat& img);
	void draw_box(cv::Mat& img, inu_object_info& info, cv::Scalar& color);
	void draw_object_id(cv::Mat& img, inu_object_info& info, cv::Scalar& color);
	cv::Mat draw_KAIST_104_object(cv::Mat& img, inu_object_info& info, cv::Scalar& color, const char* db_path = "../data/KAIST-104/DB");
};

#endif