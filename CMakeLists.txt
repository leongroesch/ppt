cmake_minimum_required(VERSION 2.6)

project(CppConv)
find_package(CUDA QUIET REQUIRED)

cuda_add_executable(cpp_conv convolution.h conv_cuda.cu)
