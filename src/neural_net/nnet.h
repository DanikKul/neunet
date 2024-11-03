#pragma once

#include <vector>
#include "layers.h"

namespace nn {
    class NeuralNet {
    public:

        NeuralNet() = default;

        NeuralNet(const std::vector<int>& config) {

            assert(!config.empty());

            for (int i = 0; i < config.size(); i++) {
                int neurons_count = config[i];
                if (i == 0) {
                    layers.push_back(layers::Layer(neurons_count));
                } else {
                    layers::Layer& prev = layers.at(layers.size() - 1);
                    layers.push_back(prev.next_layer(neurons_count));
                }
            }

            for (layers::Layer &layer: layers) {
                layer.weights.fillRandom(-.5, .5);
            }
        }

        Matrix& getOutputs() {
            return layers[layers.size() - 1].outputs;
        }

        void pass(const Matrix& input) {
            layers[0].outputs = input;
            for (size_t i = 1; i < layers.size(); i++) {
                layers::Layer& curr = layers[i];
                layers::Layer& prev = layers[i - 1];
                layers::Layer::pass(curr, prev);
            }
        }

        void backprop(const Matrix& expected) {
            Matrix& output = layers[layers.size() - 1].outputs;
            Matrix delta = output - expected;
            for (size_t i = layers.size() - 1; i > 0; i--) {
                layers::Layer& curr = layers[i];
                layers::Layer& prev = layers[i - 1];

                curr.biased += delta * -learn_rate;
                prev.weights += (prev.outputs.transpose() * delta) * -learn_rate;

                Matrix one = Matrix(prev.outputs.rows, prev.outputs.cols, 1);
                Matrix sigmoid_derivative = prev.outputs.multiply_like_value(one - prev.outputs);

                delta = (delta * prev.weights.transpose()).multiply_like_value_inplace(sigmoid_derivative);
            }
        }

        void save(const char* path) const {
            FILE* fp;
            fp = fopen(path, "w");

            fprintf(fp, "%d ", (int)layers.size());
            for (int i = 0; i < layers.size(); i++) {
                fprintf(fp, "%d ", layers[i].outputs.cols);
            }

            fprintf(fp, "\nW\n");

            for (int i = 0; i < layers.size(); i++) {
                fprintf(fp, "%d %d\n", i, layers[i].weights.rows);
                for (int j = 0; j < layers[i].weights.rows; j++) {
                    for (int k = 0; k < layers[i].weights.cols; k++) {
                        fprintf(fp, "%lf ", layers[i].weights.at(j, k));
                    }
                    fprintf(fp, "\n");
                }
            }

            fprintf(fp, "\nB\n");

            for (int i = 0; i < layers.size(); i++) {
                fprintf(fp, "%d %d\n", i, layers[i].weights.rows);
                for (int j = 0; j < layers[i].biased.rows; j++) {
                    for (int k = 0; k < layers[i].biased.cols; k++) {
                        fprintf(fp, "%lf ", layers[i].biased.at(j, k));
                    }
                    fprintf(fp, "\n");
                }
            }

            fclose(fp);
        }

        void load(const char* path) {
            FILE* fp;
            char buff[10000];
            int currLayers = 0, currLayer = 0, currRows = 0;
            fp = fopen(path, "r");
            fscanf(fp, "%d ", &currLayers);
            std::vector<int> neuronsPerLayer;
            for (int i = 0; i < currLayers; i++) {
                int n_buff = 0;
                fscanf(fp, "%d ", &n_buff);
                neuronsPerLayer.emplace_back(n_buff);
            }

            layers = std::vector<layers::Layer>();
            for (int i = 0; i < currLayers; i++) {
                int neurons_count = neuronsPerLayer[i];
                if (i == 0) {
                    layers.push_back(layers::Layer(neurons_count));
                } else {
                    layers::Layer& prev = layers.at(layers.size() - 1);
                    layers.push_back(prev.next_layer(neurons_count));
                }
            }

            neuronsPerLayer.emplace_back(0);
            fgets(buff, 10000, fp);
            for (int i = 0; i < currLayers; i++) {
                fscanf(fp, "%d %d", &currLayer, &currRows);
                Matrix m(currRows, neuronsPerLayer[i + 1]);
                for (int j = 0; j < currRows; j++) {
                    for (int k = 0; k < neuronsPerLayer[i + 1]; k++) {
                        float w_buff = 0;
                        fscanf(fp, "%f ", &w_buff);
                        m.set(j, k, w_buff);
                    }
                }
                this->layers[i].weights.copyData(m.matrix, currRows, neuronsPerLayer[i + 1]);
            }
            for (int i = 0; i < 3; i++) fgets(buff, 10000, fp);
            for (int i = 0; i < currLayers; i++) {
                fscanf(fp, "%d %d", &currLayer, &currRows);
                Matrix m(1, neuronsPerLayer[i]);
                for (int k = 0; k < neuronsPerLayer[i]; k++) {
                    float w_buff = 0;
                    fscanf(fp, "%f ", &w_buff);
                    m.set(0, k, w_buff);
                }
                this->layers[i].biased.copyData(m.matrix, 1, neuronsPerLayer[i]);
            }
        }

        T learn_rate = 0.01;
        std::vector<layers::Layer> layers;
    };
}