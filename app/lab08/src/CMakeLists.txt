cmake_minimum_required (VERSION 2.8)

set (opencv_include_dir D:/project/lib/opencv3.3/include	D:/project/lib/opencv3.3/include/opencv2)
set (opencv_lib_dir D:/project/lib/opencv3.3/x64/vc15/lib)
set (opencv_lib opencv_world330.lib)
set (opencv_libd opencv_world330d.lib)

set (src inu_generalized_hough_transform.cpp	inu_generalized_hough_transform.h
	inu_object_identification.cpp		inu_object_identification.h
	main.cpp				inu_object_info.h
)

project (2018_CV)

include_directories (${opencv_include_dir})
link_directories (${opencv_lib_dir})
add_executable (2018_CV ${src})
set(EXECUTABLE_OUTPUT_PATH "../bin")

target_link_libraries (2018_CV ${opencv_lib}) #here