#include <vector>

#include "inu_object_identification.h"

int main()
{
	// an instance of the inu_object_identification class
	inu_object_identification ioi;
	// train the knn tree from KAIST 104 DB
	// the number of model images is 104, we can decrease the number for debugging.
	ioi.train_knn_tree_from_KAIST_104(104);

	// load an input image. you can try different input images.
	char fn[1024];
	sprintf(fn, "../data/KAIST-104/Rotation/23_4.jpg");
	cv::Mat img = cv::imread(fn);

	// find objects from the input image
	// please refer to "inu_object_info.h" for the return type
	std::vector<inu_object_info> objects = ioi.find_objects(img, 1, 4);

	// draw the objects on the input image sequentially.
	for (int i = 0; i < objects.size(); ++i)
	{
		printf("%d\n", objects[i].m_id);
		cv::Mat comb = ioi.draw_KAIST_104_object(img, objects[i], cv::Scalar(0, 255, 255));
		cv::imshow("img", comb);
		sprintf(fn, "../image/%d.jpg", i);
		cv::imwrite(fn, comb);
		cv::waitKey();
	}

	return 0;
}

