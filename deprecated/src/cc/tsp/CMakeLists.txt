project(TSP)
cmake_minimum_required(VERSION 3.0)

include_directories(..)
set(CMAKE_CXX_STANDARD 11)

add_executable("TSP" ../mcts.cpp
                     main.cpp)
