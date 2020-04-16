#include "inu_object_identification.h"

#include <float.h>
#include "Inu_generalized_hough_transform.h"

using namespace std;
using namespace cv;
using namespace cv::xfeatures2d;

inu_object_identification::inu_object_identification()
{
	m_kd_tree = NULL;
}

inu_object_identification::~inu_object_identification()
{
	if (m_kd_tree)
	{
		delete m_kd_tree;
		m_kd_tree = NULL;
	}
}

int inu_object_identification::train_knn_tree_from_KAIST_104(const int nmodels, const char* db_path)
{
	// the number of models to be trained
	m_nmodels = nmodels;
	// the first model index
	m_first = 1;
	// temporary keypoints and descriptors
	vector<vector<KeyPoint>> keypoints;
	vector<Mat> descriptors;

	// clear all our member variables
	m_train_label.clear();
	m_train_center.clear();
	m_train_pt.clear();
	m_train_angle.clear();
	m_train_size.clear();

	// image file name
	char fn[1024];

	// a parameter for SURF
	int minHessian = 100;
	// SURF
	Ptr<SURF> surf = SURF::create(minHessian);

	// total number of keypoints in the model images
	int nkeypoints = 0;
	for (int i = m_first; i < nmodels+m_first; ++i)
	{
		// create the file name
		sprintf(fn, "%s/%d.jpg", db_path, i);
		// load the image
		Mat img = imread(fn);
		// convert the image to a gray image
		Mat img_gray;
		cvtColor(img, img_gray, COLOR_BGR2GRAY);

		// a train center is the center of a model image
		m_train_center.push_back(Point2f((float)(0.5*(img.cols-1)), (float)(0.5*(img.rows-1))));

		// detect the keypoints
		vector<KeyPoint> keypoint;
		surf->detect(img_gray, keypoint);
		// save the keypoints in the vector
		keypoints.push_back(keypoint);

		// describe the keypoints
		Mat descriptor;
		surf->compute(img_gray, keypoint, descriptor);
		// save the descriptor in the vector
		descriptors.push_back(descriptor);

		// count the number of keypoints
		nkeypoints += (int)(keypoint.size());
	}

	// the dimension of the descriptor vector
	m_dim = 0;
	for (int i = 0; i < m_nmodels; ++i)
	{
		m_dim = descriptors[i].cols;
		if (m_dim > 0)
			break;
	}

	// initialize the memory of the member variables
	m_train_feature = Mat::zeros(nkeypoints, m_dim, CV_32FC1);

	// save the necessary data to the member variables
	int k = 0;
	for (int i = m_first; i < nmodels+m_first; ++i)
	{
		int i1 = i - m_first;
		// number of keypoints of model i
		int nkeypoint = (int)(keypoints[i1].size());
		for (int j = 0; j < nkeypoint; ++j, ++k)
		{
			m_train_pt.push_back(keypoints[i1][j].pt);
			m_train_angle.push_back(keypoints[i1][j].angle);	// from 0 to 360 degree
			m_train_size.push_back(keypoints[i1][j].size);
			m_train_label.push_back(i);
			for (int m = 0; m < m_dim; ++m)
			{
				m_train_feature.at<float>(k, m) = descriptors[i1].at<float>(j, m);
			}
		}
	}

	descriptors.clear();
	keypoints.clear();

	// create a new k-d tree
	flann::KDTreeIndexParams indexParams;
	if (m_kd_tree)
		delete m_kd_tree;
	m_kd_tree = new flann::Index(m_train_feature, indexParams);

#if 0
	for (int i = 0; i < nkeypoints; ++i)
	{
		printf("%d\t(%f, %f)\t%f\t%f: ", m_train_label.at<int>(0, i), m_train_pos[i].x, m_train_pos[i].y, m_train_scale[i], m_train_ori[i]);
		for (int j = 0; j < dim; ++j)
		{
			printf("%f\t", m_train_feature.at<float>(i, j));
		}
		printf("\n");
	}
#endif

	printf("Training complete!\n");

	return 0;
}

Mat inu_object_identification::get_affine_transform_RANSAC(vector<Point2f>& src_pts, vector<Point2f>& dst_pts, float thresh, int max_iter)
{
	int npts = (int)(src_pts.size());
	// if the number of point pairs is three, then we don't apply RANSAC.
	if (npts == 3)
		return getAffineTransform(src_pts, dst_pts);

	// square of the threshold
	float t2 = thresh*thresh;

	// temporary vector needed for generating seed indices
	vector<double> seed_val;
	// seed indices
	vector<int> seed;

	// we want an affine transform that maximizes the number of inliers
	int max_ninliers = 0;
	// bestA will become such an affine transform.
	Mat bestA = Mat::zeros(2, 3, CV_64FC1);

	// for a given number of iterations
	for (int iter = 0; iter < max_iter; ++iter)
	{
		// randomly generate seed indices
		rand_perm(seed, seed_val, npts, 3);

		vector<Point2f> src_seed;
		vector<Point2f> dst_seed;
		for (int i = 0; i < 3; ++i)
		{
			src_seed.push_back(src_pts[seed[i]]);
			dst_seed.push_back(dst_pts[seed[i]]);
		}

		// compute an affine transform from the seed points.
		Mat A = getAffineTransform(src_seed, dst_seed);

		// count the number of inliers.
		int ninliers = 0;
		float x, y, dx, dy;
		for (int i = 0; i < npts; ++i)
		{
			x = (float)(A.at<double>(0, 0)*src_pts[i].x + A.at<double>(0, 1)*src_pts[i].y + A.at<double>(0, 2));
			y = (float)(A.at<double>(1, 0)*src_pts[i].x + A.at<double>(1, 1)*src_pts[i].y + A.at<double>(1, 2));
			dx = x - dst_pts[i].x;
			dy = y - dst_pts[i].y;

			if (dx*dx + dy*dy < t2)
				ninliers += 1;
		}
		// if the number of inliers is greater than the current maximum number of inliers
		if (ninliers > max_ninliers)
		{
			// change max_ninliers and bestA with ninliers and A
			max_ninliers = ninliers;
			bestA = A;
		}
	}

	// return the best affine transform
	return bestA;
}

// img: input image
// knn: how many matches per keypoint is allowed?
// min_num_inliers: minimum number of inliers for affine transform, typically 3.
// thresh_inlier: the square error maximally allowed by the affine transform, typically 5.
vector<inu_object_info> inu_object_identification::find_objects(cv::Mat& img, int knn, int min_num_inliers, float thresh_inlier)
{
	// a parameter for SURF
	int minHessian = 100;
	Ptr<SURF> surf = SURF::create(minHessian);

	// convert image to a gray image
	Mat img_gray;
	cvtColor(img, img_gray, COLOR_BGR2GRAY);

	// detect the keypoints
	vector<KeyPoint> keypoint;
	surf->detect(img_gray, keypoint);

	// number of keypoints
	int nkeypoints = (int)(keypoint.size());

	// describe the keypoints
	Mat descriptor;
	surf->compute(img_gray, keypoint, descriptor);

	// find k nearest neighbors to the descriptors from the input image
	vector<vector<int>> indices;
	for (int i = 0; i < nkeypoints; ++i)
	{
		vector<float> query;
		vector<int> index;
		vector<float> dist;
		for (int j = 0; j < m_dim; ++j)
			query.push_back(descriptor.at<float>(i, j));

		m_kd_tree->knnSearch(query, index, dist, knn);

		indices.push_back(index);
	}

	// collect data for hough transform
	// count the models with inliers no less than min_num_inliers
	// if the number of inliers is less than "min_num_inliers" at this stage, we don't have to go further for the model.
	vector<int> pre_vote(m_nmodels);
	// pre-computed hough bin entries per match
	vector<float> s;
	vector<float> x;
	vector<float> y;
	vector<float> o;

	double mul = CV_PI / 180; // our angle is in degree
	// initially no model has inliers
	for (int i = 0; i < m_nmodels; ++i)
		pre_vote[i] = 0;

	// for setting the hough grid, we check the ranges of the parameters
	float min_x, max_x, min_y, max_y, min_s, max_s;
	min_x = (float)(img.cols);	max_x = 0.0f;
	min_y = (float)(img.rows);	max_y = 0.0f;
	min_s = 10.0f;	max_s = 0.0f;

	// compute the hough parameters
	// please refer to our lecture note for more detail
	for (int i = 0; i < nkeypoints; ++i)
	{
		for (int k = 0; k < knn; ++k)
		{
			int ind = indices[i][k];
			if (ind > -1)
			{
				int l = m_train_label[ind];
				pre_vote[l-m_first]+=1;

				float stemp = keypoint[i].size / m_train_size[ind];
				s.push_back(stemp);
				min_s = min(stemp, min_s);
				max_s = max(stemp, max_s);

				float otemp = keypoint[i].angle - m_train_angle[ind];
				if (otemp < 0)
					otemp += 360;
				o.push_back(otemp);

				otemp *= mul;
				double co = cos(otemp);
				double so = sin(otemp);
				//float xtemp = (float)(keypoint[i].pt.x - stemp*(m_train_pt[ind].x*co - m_train_pt[ind].y*so));
				//float ytemp = (float)(keypoint[i].pt.y - stemp*(m_train_pt[ind].x*so + m_train_pt[ind].y*co));
				// theoretically, the same. practically, the below works much better.
				float xtemp = (float)(keypoint[i].pt.x + stemp*((m_train_center[l].x-m_train_pt[ind].x)*co - (m_train_center[l].y-m_train_pt[ind].y)*so));
				float ytemp = (float)(keypoint[i].pt.y + stemp*((m_train_center[l].x-m_train_pt[ind].x)*so + (m_train_center[l].y-m_train_pt[ind].y)*co));

				x.push_back(xtemp);
				y.push_back(ytemp);

				min_x = min(xtemp, min_x);
				max_x = max(xtemp, max_x);
				min_y = min(ytemp, min_y);
				max_y = max(ytemp, max_y);
			}
		}
	}

	// we further restric the parameters for robustness
	min_x = max(min_x, (float)(-0.5f*(img.cols)));
	max_x = min(max_x, (float)(1.5f*(img.cols)));
	min_y = max(min_y, (float)(-0.5*(img.rows)));
	max_y = min(max_y, (float)(1.5f*(img.rows)));
	min_s = max(min_s, 0.1f);
	max_s = min(max_s, 10.0f);

	// active object labels with no less than "min_num_inliers" inliers.
	vector<int> labels;
	// the mapping from the model index to the index of "labels"
	vector<int> label_map(m_nmodels);
	for (int i = 0; i < m_nmodels; ++i)
		label_map[i] = -1;

	int n = 0;
	for (int i = 0; i < m_nmodels; ++i)
	{
		if (pre_vote[i] >= min_num_inliers)	// we should compute an affine transform
		{
			labels.push_back(m_first + i);
			label_map[i] = n;
			++n;
		}
	}

#if 0
	printf("%f\t%f\t", min_x, max_x);
	printf("%f\t%f\t", min_y, max_y);
	printf("%f\t%f\t", min_s, max_s);
	printf("%f\t%f\t", min_o, max_o);
	printf("\n");
#endif

	// initialize the hough grid
	inu_generalized_hough_transform ight;
	ight.init(labels, min_x, max_x, min_y, max_y, min_s, max_s, (float)(img.cols / 10.0), (float)(img.cols / 10.0), 1.5f, 20.0f);

	// vote on the hough grid
	int m = 0;
	for (int i = 0; i < nkeypoints; ++i)
	{
		for (int k = 0; k < knn; ++k)
		{
			int ind = indices[i][k];
			if (ind > -1)
			{
				int l = m_train_label[ind]-m_first;
				if (label_map[l] > -1)
					ight.vote(label_map[l], x[m], y[m], s[m], o[m], Vec2i(i, k));
				++m;
			}
		}
	}

	// get the object labels with sufficient inliers and the matches that voted on the bin
	vector<vector<Vec2i>> keypoint_indices;
	vector<int> obj_label = ight.get_object_labels(keypoint_indices, min_num_inliers);

#if 0
	for (int i = 0; i < obj_label.size(); ++i)
	{
		printf("%d\n", obj_label[i]);
		for (int j = 0; j < keypoint_indices[i].size(); ++j)
			printf("(%d,%d),", keypoint_indices[i][j][0], keypoint_indices[i][j][1]);
		printf("\n");
	}
#endif

	// we are going to return this vector
	vector<inu_object_info> ret;

	// the number of bins with sufficient entries
	int nbins = (int)(obj_label.size());
	// if no object has been detected, return a null vector
	if (nbins ==0)
		return ret;

	// flag for checking if a point is an inlier to the affine transform of an object
	vector<vector<int>> inlier_flag(nkeypoints);
	for (int i = 0; i < nkeypoints; ++i)
	{
		inlier_flag[i].resize(knn);
		for (int j = 0; j < knn; ++j)
			inlier_flag[i][j] = -1;
	}
	// number of valid matches in each bin
	vector<int> num_valid(nbins);
	for (int i = 0; i < nbins; ++i)
		num_valid[i] = (int)(keypoint_indices[i].size());
	// validity of the entries in the bins. initially all valid
	vector<vector<int>> valid(nbins);
	for (int i = 0; i < nbins; ++i)
	{
		valid[i].resize(keypoint_indices[i].size());
		for (int j = 0; j < keypoint_indices[i].size(); ++j)
			valid[i][j] = 1;
	}
	vector<int> valid_key(nkeypoints);
	for (int i = 0; i < nkeypoints; ++i)
		valid_key[i] = 1;

	// square of the threshold
	float t2 = thresh_inlier*thresh_inlier;

	while (1)
	{
		// find the obj_label with the greatest number of num_valid
		int max_obj = -1;
		int max_num = 0;
		for (int i = 0; i < nbins; ++i)
		{
			if (num_valid[i] > max_num)
			{
				max_obj = i;
				max_num = num_valid[i];
			}
		}
		
		if (max_num < min_num_inliers)
			break;

		// the object label
		int max_label = obj_label[max_obj];

		// collect corresponding points
		vector<Point2f> model_pts;
		vector<Point2f> input_pts;

		for (int i = 0; i < keypoint_indices[max_obj].size(); ++i)
		{
			if (valid[max_obj][i])
			{
				input_pts.push_back(keypoint[keypoint_indices[max_obj][i][0]].pt);
				model_pts.push_back(m_train_pt[indices[keypoint_indices[max_obj][i][0]][keypoint_indices[max_obj][i][1]]]);
				valid[max_obj][i] = 0; // used once, then it is no more valid
			}
		}
		num_valid[max_obj] = 0; // used once, then it is no more valid

		// compute an affine transform
		Mat A = get_affine_transform_RANSAC(model_pts, input_pts, thresh_inlier);

		// check more inliers
		int ninliers = 0;
		float x, y, tx, ty, dx, dy;
		for (int i = 0; i < nkeypoints; ++i)
		{
			if (valid_key[i] == 0)
				continue;

			for (int j = 0; j < knn; ++j)
			{
				if (inlier_flag[i][j] > -1)
					continue;

				if (m_train_label[indices[i][j]] != max_label)
					continue;

				x = m_train_pt[indices[i][j]].x;
				y = m_train_pt[indices[i][j]].y;

				tx = (float)(A.at<double>(0, 0)*x + A.at<double>(0, 1)*y + A.at<double>(0, 2));
				ty = (float)(A.at<double>(1, 0)*x + A.at<double>(1, 1)*y + A.at<double>(1, 2));

				dx = tx - keypoint[i].pt.x;
				dy = ty - keypoint[i].pt.y;

				if (dx*dx + dy*dy<t2)
				{
					inlier_flag[i][j] = max_label;
					valid_key[i] = 0;
					++ninliers;
				}
			}
		}

		for (int i = 0; i < nbins; ++i)
		{
			// because they are already invalid.
			//if (i == max_obj)
			//	continue;

			// the inliers are now invalid, no more use.
			for (int j = 0; j < keypoint_indices[i].size(); ++j)
			{
				if (valid[i][j] == 0)
					continue;

				if (valid_key[keypoint_indices[i][j][0]]==0 || inlier_flag[keypoint_indices[i][j][0]][keypoint_indices[i][j][1]] > -1)
				{
					valid[i][j] = 0;
					num_valid[i]--;
				}
			}
		}

		if (ninliers > min_num_inliers)
		{
			inu_object_info info;
			info.m_id = max_label;
			info.m_A = A;
			x = m_train_center[max_label - m_first].x;
			y = m_train_center[max_label - m_first].y;
			tx = (float)(A.at<double>(0, 0)*x + A.at<double>(0, 1)*y + A.at<double>(0, 2));
			ty = (float)(A.at<double>(1, 0)*x + A.at<double>(1, 1)*y + A.at<double>(1, 2));
			info.m_center = Point2f(tx, ty);

			ret.push_back(info);
		}
	}

	return ret;
}

void inu_object_identification::draw_object_id(Mat& img, inu_object_info& info, Scalar& color)
{
	char text[1024];
	sprintf(text, "%d", info.m_id);
	putText(img, text, info.m_center, FONT_HERSHEY_SIMPLEX, 1, color, 2);
	drawMarker(img, info.m_center, color, 0, 20, 2);
}

void inu_object_identification::draw_box(Mat& img, inu_object_info& info, Scalar& color)
{
	Point2f pt[4];
	float x, y;

	x = 0; y = 0;
	pt[0].x = (float)(info.m_A.at<double>(0, 0)*x + info.m_A.at<double>(0, 1)*y + info.m_A.at<double>(0, 2));
	pt[0].y = (float)(info.m_A.at<double>(1, 0)*x + info.m_A.at<double>(1, 1)*y + info.m_A.at<double>(1, 2));

	x = 2 * m_train_center[info.m_id - m_first].x; y = 0;
	pt[1].x = (float)(info.m_A.at<double>(0, 0)*x + info.m_A.at<double>(0, 1)*y + info.m_A.at<double>(0, 2));
	pt[1].y = (float)(info.m_A.at<double>(1, 0)*x + info.m_A.at<double>(1, 1)*y + info.m_A.at<double>(1, 2));

	x = 0; y = 2 * m_train_center[info.m_id - m_first].y;
	pt[3].x = (float)(info.m_A.at<double>(0, 0)*x + info.m_A.at<double>(0, 1)*y + info.m_A.at<double>(0, 2));
	pt[3].y = (float)(info.m_A.at<double>(1, 0)*x + info.m_A.at<double>(1, 1)*y + info.m_A.at<double>(1, 2));

	x = 2 * m_train_center[info.m_id - m_first].x; y = 2 * m_train_center[info.m_id - m_first].y;
	pt[2].x = (float)(info.m_A.at<double>(0, 0)*x + info.m_A.at<double>(0, 1)*y + info.m_A.at<double>(0, 2));
	pt[2].y = (float)(info.m_A.at<double>(1, 0)*x + info.m_A.at<double>(1, 1)*y + info.m_A.at<double>(1, 2));

	line(img, pt[0], pt[1], color, 2);
	line(img, pt[1], pt[2], color, 2);
	line(img, pt[2], pt[3], color, 2);
	line(img, pt[3], pt[0], color, 2);
}

Mat inu_object_identification::draw_KAIST_104_object(Mat& img, inu_object_info& info, Scalar& color, const char* db_path)
{
	Mat warp = get_warped_KAIST_104_model_image(info, img, db_path);
	Mat comb = get_combined_image(warp, img);
	draw_box(comb, info, color);
	draw_object_id(comb, info, color);

	return comb;
}

Mat inu_object_identification::get_warped_KAIST_104_model_image(inu_object_info& info, Mat& img, const char* db_path)
{
	char fn[1024];
	sprintf(fn, "%s/%d.jpg", db_path, info.m_id);
	Mat model_img = imread(fn);
	Mat warp_img;
	warpAffine(model_img, warp_img, info.m_A, img.size());

	return warp_img;
}

Mat inu_object_identification::get_combined_image(Mat& warped, Mat& img)
{
	int width = img.cols;
	int height = img.rows;

	Mat combined(height, width, CV_8UC3);

	for (int i = 0; i < height; ++i)
	{
		for (int j = 0; j < width; ++j)
		{
			for (int k = 0; k < 3; ++k)
				combined.at<Vec3b>(i, j)[k] = (uchar)(0.5*warped.at<Vec3b>(i, j)[k] + 0.5*img.at<Vec3b>(i, j)[k]);
		}
	}

	return combined;
}
