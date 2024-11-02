#pragma once

#define ACT_FUNC_LINEAR 0
#define ACT_FUNC_SIGMOID 1
#define ACT_FUNC_TANH 2

#include <cmath>
namespace activation {
    double sigmoid(double x) {
        return 1 / (1 + exp(-x));
    }
}