cmake_minimum_required (VERSION 2.8)

set (opencv_include_dir C:/project/lib/opencv3.3/include	C:/project/lib/opencv3.3/include/opencv2)
set (opencv_lib_dir C:/project/lib/opencv3.3/x64/vc15/lib)
set (opencv_lib debug opencv_world330d.lib optimized opencv_world330.lib)

set (k-means_rgb_src k-means_rgb.cpp)
set (k-means_xyrgb_src k-means_xyrgb.cpp)
set(mean_shift_src mean_shift.cpp)

project (k-means_rgb)

include_directories (${opencv_include_dir})
link_directories (${opencv_lib_dir})
add_executable (k-means_rgb ${k-means_rgb_src})
set(EXECUTABLE_OUTPUT_PATH "../bin")

target_link_libraries (k-means_rgb ${opencv_lib})

project (k-means_xyrgb)

include_directories (${opencv_include_dir})
link_directories (${opencv_lib_dir})
add_executable (k-means_xyrgb ${k-means_xyrgb_src})
set(EXECUTABLE_OUTPUT_PATH "../bin")

target_link_libraries (k-means_xyrgb ${opencv_lib})

project (mean_shift)

include_directories (${opencv_include_dir})
link_directories (${opencv_lib_dir})
add_executable (mean_shift ${mean_shift_src})
set(EXECUTABLE_OUTPUT_PATH "../bin")

target_link_libraries (mean_shift ${opencv_lib})