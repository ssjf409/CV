cmake_minimum_required (VERSION 2.8)

set (opencv_include_dir D:/project/lib/opencv3.3/include	D:/project/lib/opencv3.3/include/opencv2)
set (opencv_lib_dir D:/project/lib/opencv3.3/x64/vc15/lib)
set (opencv_lib debug opencv_world330d.lib optimized opencv_world330.lib)

set (
        face_src face_detection.cpp
    )

set (
        pedestrian_src pedestrian_detection.cpp
    )

set (
        text_src text_detection.cpp
    )

project (face)

include_directories (${opencv_include_dir})
link_directories (${opencv_lib_dir})
add_executable (face ${face_src})
set(EXECUTABLE_OUTPUT_PATH "../bin")

target_link_libraries (face ${opencv_lib})

project (pedestrian)

include_directories (${opencv_include_dir})
link_directories (${opencv_lib_dir})
add_executable (pedestrian ${pedestrian_src})
set(EXECUTABLE_OUTPUT_PATH "../bin")

target_link_libraries (pedestrian ${opencv_lib})

project (text)

include_directories (${opencv_include_dir})
link_directories (${opencv_lib_dir})
add_executable (text ${text_src})
set(EXECUTABLE_OUTPUT_PATH "../bin")

target_link_libraries (text ${opencv_lib})