#pragma once
#include "../matrix/matrix.h"

#define LAYER_INPUT 0
#define LAYER_HIDDEN 1
#define LAYER_OUTPUT 2

using namespace matrix;

namespace layers {
    class Layer {
    public:
        explicit Layer(int n) {
            outputs = Matrix(1, n);
            biased = Matrix(1, n);
            weights = Matrix(0, 0);
            outputs.alloc();
            biased.alloc();
            outputs.fillVal(0);
            biased.fillVal(0);
        }

        Layer(Layer&& other) noexcept :
                outputs(std::move(other.outputs)),
                biased(std::move(other.biased)),
                weights(std::move(other.weights))
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