project(MCTS)
cmake_minimum_required(VERSION 3.0)

add_subdirectory(src/cc/chess)
add_subdirectory(src/cc/connect4)

if (${BUILD_Deprecated})
 add_subdirectory(deprecated/src/cc/tsp)
 add_subdirectory(deprecated/src/cc/hearts)
endif()
