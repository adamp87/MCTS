project(Hearts)
cmake_minimum_required(VERSION 2.8)

find_package(CUDA REQUIRED)

set(CMAKE_CXX_STANDARD 11)
#set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS} "-G" ) # debug
set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS} "-lineinfo" ) # compiler info
set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS} "-std=c++11" ) # fix for linux
#set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS} "-gencode arch=compute_30,code=sm_30") # compile for architecture
#set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS} "-gencode arch=compute_50,code=sm_50") # compile for architecture

add_executable(${PROJECT_NAME} src/flownetwork.cpp
                               src/mcts.cpp
                               src/main.cpp)

CUDA_ADD_EXECUTABLE("${PROJECT_NAME}GPU" src/mcts.cu
                                         src/main.cpp)
