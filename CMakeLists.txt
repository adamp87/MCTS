project(MCTS)
cmake_minimum_required(VERSION 3.0)

find_package(CUDA QUIET)

set(CMAKE_CXX_STANDARD 11)

add_executable("TSP" src/mcts.cpp
                     src/main_tsp.cpp)

add_executable("Hearts" src/mcts.cpp
                     src/flownetwork.cpp
                     src/main_hearts.cpp)

add_executable("Chess" src/mcts.cpp
                       src/main_chess.cpp)
target_link_libraries("Chess" PUBLIC zmq.so)

add_executable("Connect4" src/mcts.cpp
                          src/main_connect4.cpp)
target_link_libraries("Connect4" PUBLIC zmq.so)

if (CUDA_FOUND)
    message (STATUS "Found CUDA${CUDA_VERSION_STRING}")
    #set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS} "-G" ) # debug
    set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS} "-lineinfo" ) # compiler info
    set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS} "-std=c++11" ) # fix for linux
    #set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS} "-gencode arch=compute_50,code=sm_50") # compile for architecture

    CUDA_ADD_EXECUTABLE("HeartsGPU" src/mcts.cu
                                    src/main_hearts.cpp)
else()
    message(STATUS "No CUDA was found, only compiling CPU version")
endif()
