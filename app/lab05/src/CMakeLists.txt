cmake_minimum_required (VERSION 2.8)

set (opencv_include_dir C:/project/lib/opencv3.3/include	C:/project/lib/opencv3.3/include/opencv2)
set (opencv_lib_dir C:/project/lib/opencv3.3/x64/vc15/lib)
set (opencv_lib debug opencv_world330d.lib optimized opencv_world330.lib)

set (
        harris_src harris_corner.cpp
    )

set (
        line_src hough_line.cpp
    )

set (
        circle_src hough_circle.cpp
    )

project (hough_line)

include_directories (${opencv_include_dir})
link_directories (${opencv_lib_dir})
add_executable (hough_line ${line_src})
set(EXECUTABLE_OUTPUT_PATH "../bin")

target_link_libraries (hough_line ${opencv_lib})

project (hough_circle)

include_directories (${opencv_include_dir})
link_directories (${opencv_lib_dir})
add_executable (hough_circle ${circle_src})
set(EXECUTABLE_OUTPUT_PATH "../bin")

target_link_libraries (hough_circle ${opencv_lib})

project (harris_corner)

include_directories (${opencv_include_dir})
link_directories (${opencv_lib_dir})
add_executable (harris_corner ${harris_src})
set(EXECUTABLE_OUTPUT_PATH "../bin")

target_link_libraries (harris_corner ${opencv_lib})