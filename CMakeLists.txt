cmake_minimum_required(VERSION 2.6)

project(CppConv)
find_package(CUDA QUIET REQUIRED)

cuda_add_executable(cpp_conv conv_cuda.cu)
