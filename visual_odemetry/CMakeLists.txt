cmake_minimum_required(VERSION 3.5)


set(CMAKE_BUILD_TYPE "Release")
set(CMAKE_CXX_FLAGS "-std=c++11")

set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED TRUE)

include_directories("/usr/local/include/eigen3")

find_package(Sophus REQUIRED)
include_directories(${Sophus_INCLUDE_DIRS})
find_package(fmt REQUIRED)
include_directories(${fmt_INCLUDE_DIRS})

find_package(OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})
#include_directories("/home/pinhao/pinhao/opencv/include/opencv2")

#find_package(PCL 1.12 REQUIRED COMPONENTS common io)
#find_package(PCL 1.12 REQUIRED)
#include_directories(${PCL_INCLUDE_DIRS})
#link_directories(${PCL_LIBRARY_DIRS})

#add_definitions(${PCL_DEFINITIONS})
add_executable(output main.cpp)
target_link_libraries(output ${OpenCV_LIBS})
target_link_libraries(output fmt)
#target_link_libraries(output ${PCL_LIBRARIES})
