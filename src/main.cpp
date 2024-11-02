#include "neural_net/nnet.h"
#include "dataset/dataset.h"

using namespace layers;

int main() {


//    -------------TRAIN MODEL-------------

    nn::NeuralNet net({ 2, 3, 3, 1 });

    dataset::Dataset trainset("/Users/dankulakovich/CLionProjects/nn/train/XOR/train.dset");
    dataset::train(net, trainset, 1000000, false, 0.000001);

    net.save("/Users/dankulakovich/CLionProjects/nn/models/XOR.model");

//    -------------LOAD AND TEST-------------

//    nn::NeuralNet net1;
//    dataset::Dataset testset("/Users/dankulakovich/CLionProjects/nn/test/XOR/test.dset");
//    net1.load("/Users/dankulakovich/CLionProjects/nn/models/XOR.model");
//    dataset::test(net1, testset, false, 0.01);

    return 0;
}
