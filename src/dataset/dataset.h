#pragma once

#include <cstdlib>
#include "../neural_net/nnet.h"
#include "../neural_net/errors.h"

using namespace nn;
using namespace matrix;

namespace dataset {
    class Dataset {
    public:

        // Input and output matrices are separated
        Dataset(Matrix& input, Matrix& output) {
            assert(input.rows == output.rows);
            this->input.copyData(input.matrix, input.rows, input.cols);
            this->output.copyData(output.matrix, output.rows, output.cols);
        }

        // Input and output matrices are contained in one, in format "i1 i2 i3 i4 i5 o1 o2 o3" in matrix object
        Dataset(Matrix& data, int i_len, int o_len) {
            this->input = Matrix(data.rows, i_len);
            this->output = Matrix(data.rows, o_len);
            for (int i = 0; i < data.rows; i++) {
                for (int j = 0; j < data.cols; j++) {
                    if (j < i_len) {
                        this->input.set(i, j, data.at(i, j));
                    } else {
                        this->output.set(i, j - i_len, data.at(i, j));
                    }
                }
            }
        }

        // Input and output matrices are contained in one, in format "i1 i2 i3 i4 i5 o1 o2 o3" in file
        Dataset(const char* path) {
            FILE* fp;
            int cases = 0, i_len = 0, o_len = 0;
            fp = fopen(path, "r");
            fscanf(fp, "%d %d %d\n", &cases, &i_len, &o_len);
            Matrix data(cases, i_len + o_len);
            for (int i = 0; i < cases; i++) {
                for (int j = 0; j < i_len + o_len; j++) {
                    float buff = 0;
                    fscanf(fp, "%f", &buff);
                    data.set(i, j, buff);
                }
            }
            this->input = Matrix(data.rows, i_len);
            this->output = Matrix(data.rows, o_len);
            for (int i = 0; i < data.rows; i++) {
                for (int j = 0; j < data.cols; j++) {
                    if (j < i_len) {
                        this->input.set(i, j, data.at(i, j));
                    } else {
                        this->output.set(i, j - i_len, data.at(i, j));
                    }
                }
            }
        }

        int count() const {
            return output.rows;
        }

        Matrix get_input(int index) const {
            Matrix m(1, input.cols);
            for (int i = 0; i < input.cols; i++) {
                m.set(0, i, input.at(index, i));
            }
            return m;
        }

        Matrix get_output(int index) const {
            Matrix m(1, output.cols);
            for (int i = 0; i < output.cols; i++) {
                m.set(0, i, output.at(index, i));
            }
            return m;
        }

    private:
        Matrix input;
        Matrix output;
    };


    void train(NeuralNet &nn, const Dataset &dataset, int epoch, bool verbose = false, float etalonError = -1) {
        bool exit_train = false;
        for (int _ = 0; _ < epoch; _++) {
            if (exit_train) {
                break;
            }
            for (int i = 0; i < dataset.count(); i++) {
                Matrix input = dataset.get_input(i);
                Matrix expected = dataset.get_output(i);

                nn.pass(input);
                Matrix& outputs = nn.getOutputs();
                nn.backprop(expected);

                float err = errors::MSE(outputs, expected);
                if (verbose) {
                    printf("Epoch: %d, error: %.4f\n", _, err);
                }
                if (err <= etalonError and etalonError != -1) {
                    exit_train = true;
                    break;
                }
            }
        }
    }

    void test(NeuralNet &nn, const Dataset &dataset, bool verbose = false, float accuracy = 0.2) {
        int failed = 0;
        for (int i = 0; i < dataset.count(); i++) {
            Matrix input = dataset.get_input(i);
            Matrix expected = dataset.get_output(i);

            nn.pass(input);
            Matrix& outputs = nn.getOutputs();
            nn.backprop(expected);

            float err = errors::MSE(outputs, expected);
            if (err > accuracy) {
                failed++;
            }

            if (verbose) {
                printf("Test case: %d, error: %.4f\n", i, err);
            }
        }
        if (failed > 0) {
            printf("Failed %d tests\n", failed);
        } else {
            printf("All tests succeeded\n");
        }
    }
}