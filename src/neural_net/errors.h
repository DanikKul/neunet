#pragma once

#define ACT_FUNC_LINEAR 0
#define ACT_FUNC_SIGMOID 1
#define ACT_FUNC_TANH 2

#include <cmath>
#include "../matrix/matrix.h"

namespace errors {
    inline float MSE(const matrix::Matrix& actual, const matrix::Matrix& expected) {
        float sum = 0;
        for (int i = 0; i < actual.cols; i++) {
            sum += (float)pow(expected.at(0, i) - actual.at(0, i), 2);
        }
        return sum / (float)actual.cols;
    }

    inline float rootMSE(const matrix::Matrix& actual, const matrix::Matrix& expected) {
        return (float)std::sqrt(MSE(actual, expected));
    }

    inline float arctanSE(const matrix::Matrix& actual, const matrix::Matrix& expected) {
        float sum = 0;
        for (int i = 0; i < actual.cols; i++) {
            sum += (float)pow(std::atan(expected.at(0, i) - actual.at(0, i)), 2);
        }
        return sum / (float)actual.cols;
    }
}
