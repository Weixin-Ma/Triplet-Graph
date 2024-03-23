cmake_minimum_required(VERSION 2.8.3)
project(tripletgraph)

set(CMAKE_BUILD_TYPE "Release")
set(CMAKE_CXX_FLAGS "-std=c++14")
set(CMAKE_CXX_FLAGS_RELEASE "-O3 -Wall -g -pthread")


find_package(catkin REQUIRED COMPONENTS
  roscpp
  rospy
  std_msgs
  sensor_msgs
)

find_package(OpenCV REQUIRED)
include_directories(include ${OpenCV_INCLUDE_DIRS})
link_directories(${OpenCV_LIB_DIR})


find_package(PkgConfig REQUIRED)
pkg_check_modules(YAMLCPP yaml-cpp REQUIRED)
if(YAMLCPP_VERSION VERSION_GREATER "0.5.0")
    add_definitions(-DHAVE_YAMLCPP_GT_0_5_0)
endif()

find_package(OpenMP)
if (OPENMP_FOUND)
    set (CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
    set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
    set (CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} ${OpenMP_EXE_LINKER_FLAGS}")
endif()

find_package(PCL REQUIRED)
include_directories(include ${PCL_INCLUDE_DIRS})
link_directories(${PCL_LIBRARY_DIRS})

find_package(Ceres REQUIRED)
include_directories(${CERES_INCLUDE_DIRS})

catkin_package(
#  INCLUDE_DIRS include
#  LIBRARIES tripletgraph
#  CATKIN_DEPENDS roscpp rospy std_msgs
#  DEPENDS system_lib
)

include_directories(
# include
  ${catkin_INCLUDE_DIRS}
)

add_executable(tripletgraph_part_sequence_node src/tripletgraph_part_sequence_node.cpp src/tripletgraph.cpp)   
target_link_libraries(tripletgraph_part_sequence_node ${catkin_LIBRARIES} ${OpenCV_LIBS} ${PCL_LIBRARIES} ${YAMLCPP_LIBRARIES} ${CERES_LIBRARIES})


add_executable(tripletgraph_entire_sequence_node src/tripletgraph_entire_sequence_node.cpp src/tripletgraph.cpp)    
target_link_libraries(tripletgraph_entire_sequence_node ${catkin_LIBRARIES} ${OpenCV_LIBS} ${PCL_LIBRARIES} ${YAMLCPP_LIBRARIES} ${CERES_LIBRARIES})


add_executable(tripletgraph_single_pair_node src/tripletgraph_single_pair_node.cpp src/tripletgraph.cpp)    
target_link_libraries(tripletgraph_single_pair_node ${catkin_LIBRARIES} ${OpenCV_LIBS} ${PCL_LIBRARIES} ${YAMLCPP_LIBRARIES} ${CERES_LIBRARIES})