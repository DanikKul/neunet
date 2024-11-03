cmake_minimum_required(VERSION 3.27)
project(nn)

set(CMAKE_CXX_STANDARD 17)

add_executable(nn src/main.cpp
        src/neural_net/errors.h
        src/neural_net/layers.h
        src/neural_net/nnet.h
        src/dataset/dataset.h
        src/matrix/matrix.h
)

find_package(OpenMP)
if(OpenMP_CXX_FOUND)
    target_link_libraries(nn PUBLIC OpenMP::OpenMP_CXX)
endif()

target_compile_options(nn PUBLIC -fopenmp -Ofast)