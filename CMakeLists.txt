cmake_minimum_required(VERSION 2.6)

project(CppConv)

add_subdirectory("/usr/src/gtest" ${CMAKE_BINARY_DIR}/gtest)
enable_testing()
include_directories(${GTEST_INCLUDE_DIRS})

find_package(CUDA QUIET REQUIRED)

cuda_add_executable(cpp_conv array_management.h convolution.h test_cases.h conv_cuda.cu)
target_link_libraries(cpp_conv gtest gtest_main pthread)


add_test(AllTests test)
