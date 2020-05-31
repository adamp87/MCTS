project(Connect4)

#find_package(ZeroMQ)
if (MSVC)
    include_directories(${ZeroMQ_DIR}/include)
    set(ZeroMQ_Library ${ZeroMQ_DIR}/lib/libzmq-v141-mt-4_3_2.lib)
    message(${ZeroMQ_Library})
else()
    set(ZeroMQ_Library zmq.so)
endif()

include_directories(..)
set(CMAKE_CXX_STANDARD 11)

add_executable("Connect4" main.cpp)
target_link_libraries("Connect4" PUBLIC ${ZeroMQ_Library})
