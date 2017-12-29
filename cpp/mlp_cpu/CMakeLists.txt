cmake_minimum_required(VERSION 3.0)

project(mlp_mxnet)

set(CMAKE_BUILD_TYPE RelWithDebInfo)
#set(CMAKE_CXX_FLAGS "-O3 -fsanitize=address")
set(CMAKE_CXX_FLAGS "-O3")

add_definitions(-DMSHADOW_USE_CBLAS -DMSHADOW_USE_SSE=0 -Wall)
set(MXNET_ROOT /home/ec2-user/environment/source/incubator-mxnet/ CACHE STRING "mxnet")
include_directories(
  /usr/local/cuda-8.0/include
  ${MXNET_ROOT}include
  ${MXNET_ROOT}dmlc-core/include
  ${MXNET_ROOT}mshadow
  ${MXNET_ROOT}nnvm/include
  ${MXNET_ROOT}dlpack/include
  ${MXNET_ROOT}cpp-package/include
  )

link_directories(
  ${MXNET_ROOT}lib/
  /usr/local/cuda-8.0/lib64/
  /usr/local/lib64
  )

set(CMAKE_EXPORT_COMPILE_COMMANDS on)
set(CMAKE_CXX_FLAGS "-std=c++11")
set(CMAKE_CXX_STANDARD 11)

add_library(mlp_mxnet mlp.cpp)
target_link_libraries(mlp_mxnet mxnet)

add_executable(iris_mxnet iris_mxnet.cpp)
target_link_libraries(iris_mxnet mlp_mxnet)

file(COPY
  ${CMAKE_SOURCE_DIR}/iris_train_data.csv
  ${CMAKE_SOURCE_DIR}/iris_train_label.csv
  ${CMAKE_SOURCE_DIR}/iris_test_data.csv
  ${CMAKE_SOURCE_DIR}/iris_test_label.csv
  DESTINATION ${CMAKE_BINARY_DIR})
