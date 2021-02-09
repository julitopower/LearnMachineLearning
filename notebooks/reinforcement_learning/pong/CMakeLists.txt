cmake_minimum_required(VERSION 3.12)
project(pong_gym)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
set(CMAKE_CXX_STANDARD 17)

add_executable(app main.cpp)

option(PYTHON_BINDINGS "Build Python3 bindings" ON)

if(PYTHON_BINDINGS)
  message(STATUS "Python3 bindings build ON")
  find_package(Python3 REQUIRED COMPONENTS Interpreter Development)
  include_directories(${Python3_INCLUDE_DIRS})  

  add_library(pong_py_bindings SHARED pong_python.cpp)
  target_link_libraries(pong_py_bindings ${PYTHON_LIBRARY})
  set_target_properties(pong_py_bindings
    PROPERTIES
      PREFIX ""
      OUTPUT_NAME "pong"
      LINKER_LANGUAGE CXX
    )
endif(PYTHON_BINDINGS)  

add_library(pong SHARED game.cpp)
