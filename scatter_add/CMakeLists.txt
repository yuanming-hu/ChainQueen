cmake_minimum_required(VERSION 3.8)
project(scatter_inc)

set(CMAKE_CXX_FLAGS "-shared -fPIC -O2 -D_GLIBCXX_USE_CXX11_ABI=0")

execute_process(
        COMMAND python -c "import tensorflow as tf;import sys; sys.stdout.write(tf.sysconfig.get_include())"
        OUTPUT_VARIABLE TF_INCLUDE_DIRS)
message(TF include: ${TF_INCLUDE_DIRS})
set(CMAKE_CXX_STANDARD 14)
include_directories(${TF_INCLUDE_DIRS})

set(SOURCE_FILES scatter_inc.cpp scatter_inc_grad.cpp scatter_inc_grad_grad.cpp)
add_library(scatter_inc SHARED ${SOURCE_FILES})
