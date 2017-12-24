cmake_minimum_required(VERSION 3.0)

project(iris_mxnet)

set(CMAKE_BUILD_TYPE RelWithDebInfo)
#set(CMAKE_CXX_FLAGS "-O3 -fsanitize=address")
set(CMAKE_CXX_FLAGS "-O3")

add_definitions(-DMSHADOW_USE_CBLAS -DMSHADOW_USE_SSE=0)
include_directories(
  /usr/local/cuda-8.0/include
  /media/nvidia/development/sources/incubator-mxnet/include
  /media/nvidia/development/sources/incubator-mxnet/dmlc-core/include
  /media/nvidia/development/sources/incubator-mxnet/mshadow
  /media/nvidia/development/sources/incubator-mxnet/nnvm/include
  /media/nvidia/development/sources/incubator-mxnet/dlpack/include
  /media/nvidia/development/sources/incubator-mxnet/cpp-package/include
  )

link_directories(
  /media/nvidia/development/sources/incubator-mxnet/lib/
  /usr/local/cuda-8.0/lib64/
  )

set(CMAKE_EXPORT_COMPILE_COMMANDS on)
set(CMAKE_CXX_STANDARD 11)

add_executable(iris_mxnet iris_mxnet.cpp)
target_link_libraries(iris_mxnet mxnet cuda cudart)

add_executable(reader reader.cpp)
target_link_libraries(reader mxnet)

file(COPY
  ${CMAKE_SOURCE_DIR}/iris_train_data.csv
  ${CMAKE_SOURCE_DIR}/iris_train_label.csv
  ${CMAKE_SOURCE_DIR}/iris_test_data.csv
  ${CMAKE_SOURCE_DIR}/iris_test_label.csv  
  DESTINATION ${CMAKE_BINARY_DIR})
