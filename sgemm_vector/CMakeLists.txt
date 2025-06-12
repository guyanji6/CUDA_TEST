cmake_minimum_required(VERSION 3.18)
project(MyCUDAProject)
enable_language(CUDA)
add_executable(my_program sgemm.cu)