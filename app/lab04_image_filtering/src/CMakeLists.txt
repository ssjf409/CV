cmake_minimum_required (VERSION 2.8)

set (opencv_include_dir C:/project/lib/opencv3.3/include	C:/project/lib/opencv3.3/include/opencv2)
set (opencv_lib_dir C:/project/lib/opencv3.3/x64/vc15/lib)
set (opencv_lib debug opencv_world330d.lib optimized opencv_world330.lib)

set (
        src main.cpp
    )

project (image_filtering)

include_directories (${opencv_include_dir})
link_directories (${opencv_lib_dir})
add_executable (image_filtering ${src})
set(EXECUTABLE_OUTPUT_PATH "../bin")

target_link_libraries (image_filtering ${opencv_lib})