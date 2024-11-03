#pragma once
#include "../matrix/matrix.h"

using namespace matrix;

namespace layers {
    class Layer {
    public:
        explicit Layer(int n) {
            outputs = Matrix(1, n);
            biased = Matrix(1, n);
            weights = Matrix(0, 0);
        }

        Layer(Layer&& other) noexcept :
          outputs(other.outputs),
          biased(other.biased),
          weights(other.weights)
        {}

        Layer next_layer(int neuron_count) {
            Layer next(neuron_count);
            this->weights = Matrix(this->outputs.cols, next.outputs.cols);
            return next;
        }

        static void pass(Layer& curr, Layer& prev) {
            curr.outputs = (
                    (prev.outputs * prev.weights) += curr.biased
            ).sigmoid();
        }

        Matrix outputs;
        Matrix weights;
        Matrix biased;
    };
}