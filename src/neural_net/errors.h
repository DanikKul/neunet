#pragma once

#define ACT_FUNC_LINEAR 0
#define ACT_FUNC_SIGMOID 1
#define ACT_FUNC_TANH 2

#include <cmath>
#include "../matrix/matrix.h"

namespace errors {
    float MSE(const matrix::Matrix& actual, const matrix::Matrix& expected) {
        double sum = 0;
        for (int i = 0; i < actual.cols; i++) {
            sum += pow(expected.at(0, i) - actual.at(0, i), 2);
        }
        return sum / actual.cols;
    }

    float rootMSE(const matrix::Matrix& actual, const matrix::Matrix& expected) {
        return sqrt(MSE(actual, expected));
    }

    float arctanSE(const matrix::Matrix& actual, const matrix::Matrix& expected) {
        double err = 0;
        double sum = 0;
        for (int i = 0; i < actual.cols; i++) {
            sum += pow(atan(expected.at(0, i) - actual.at(0, i)), 2);
        }
        return sum / actual.cols;
    }
}
