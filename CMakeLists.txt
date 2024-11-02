cmake_minimum_required(VERSION 3.27)
project(nn)

set(CMAKE_CXX_STANDARD 17)

add_executable(nn src/main.cpp
        src/neural_net/actfunc.h
        src/neural_net/errors.h
        src/neural_net/neurons.h
        src/neural_net/layers.h
        src/neural_net/nnet.h
        src/dataset/dataset.h
        src/matrix/matrix.h)
