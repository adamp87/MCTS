project(Hearts)
cmake_minimum_required(VERSION 3.0)

find_package(CUDA QUIET)

include_directories(..)
set(CMAKE_CXX_STANDARD 11)

add_executable("Hearts" ../mcts.cpp
                        flownetwork.cpp
                        main.cpp)

if (CUDA_FOUND)
    message (STATUS "Found CUDA${CUDA_VERSION_STRING}")
    #set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS} "-G" ) # debug
    set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS} "-lineinfo" ) # compiler info
    set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS} "-std=c++11" ) # fix for linux
    #set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS} "-gencode arch=compute_50,code=sm_50") # compile for architecture

    CUDA_ADD_EXECUTABLE("HeartsGPU" ../mcts.cu
                                    main.cpp)
else()
    message(STATUS "No CUDA was found, only compiling CPU version")
endif()
