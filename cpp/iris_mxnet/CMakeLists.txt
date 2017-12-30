cmake_minimum_required(VERSION 2.8)

project(iris_mxnet)

# Set submodules path, and look for MxNet
set (CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} ${CMAKE_CURRENT_SOURCE_DIR}/../../cmake/Modules/)
find_package(MxNet REQUIRED)

# Set build type and force O3 manually
set(CMAKE_BUILD_TYPE RelWiQCthDebInfo)
set(CMAKE_CXX_FLAGS "-O3")
set(CMAKE_CXX_FLAGS "-std=c++11")
set(CMAKE_CXX_STANDARD 11)
set(CMAKE_EXPORT_COMPILE_COMMANDS on)

# Setup include and lib paths. A few definitions are needed in order to use
# MxNet headers.
add_definitions(-DMSHADOW_USE_CBLAS -DMSHADOW_USE_SSE=0)
include_directories(
  /usr/local/cuda-8.0/include
  ${MxNet_INCLUDE_DIRS}
  )

link_directories(
  /usr/local/cuda-8.0/lib64/
  /usr/local/lib64
  )

# First demo executable, REQUIRES CUDA
add_executable(iris_mxnet iris_mxnet.cpp)
target_link_libraries(iris_mxnet ${MxNet_LIB} cuda cudart)

# Second demo executable, only requires MxNet
add_executable(reader reader.cpp)
target_link_libraries(reader ${MxNet_LIB})

# Copy files for demo executables to work
file(COPY
  ${CMAKE_CURRENT_SOURCE_DIR}/iris_train_data.csv
  ${CMAKE_CURRENT_SOURCE_DIR}/iris_train_label.csv
  ${CMAKE_CURRENT_SOURCE_DIR}/iris_test_data.csv
  ${CMAKE_CURRENT_SOURCE_DIR}/iris_test_label.csv  
  DESTINATION ${CMAKE_BINARY_DIR})
