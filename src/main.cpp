#include <chrono>
#include "neural_net/nnet.h"
#include "dataset/dataset.h"

using namespace std::chrono;

int main() {


//    -------------TRAIN MODEL-------------

//    nn::NeuralNet net({ 2, 3, 3, 1 });
//
//    dataset::FileDataset trainset("/Users/dankulakovich/CLionProjects/nn/train/XOR/train.dset");
//    dataset::train(net, trainset, 1000000, false, 0.000001);
//
//    net.save("/Users/dankulakovich/CLionProjects/nn/models/XOR.model");

//    -------------LOAD AND TEST-------------

//    nn::NeuralNet net1;
//    dataset::FileDataset testset("/Users/dankulakovich/CLionProjects/nn/test/XOR/test.dset");
//    net1.load("/Users/dankulakovich/CLionProjects/nn/models/XOR.model");
//    dataset::test(net1, testset, false, 0.01);

//    -------------MNIST DATASET-------------
    printf("Started MNIST dataset loading\n");
    auto start = high_resolution_clock::now();
    const char* train_images_path = "/Users/dankulakovich/CLionProjects/nn/train/MNIST/train-images.idx3-ubyte";
    const char* train_labels_path = "/Users/dankulakovich/CLionProjects/nn/train/MNIST/train-labels.idx1-ubyte";
    dataset::MNISTDataset mnist(train_images_path, train_labels_path, 60000);
    auto stop = high_resolution_clock::now();
    auto duration = duration_cast<milliseconds>(stop - start);
    printf("Ended MNIST dataset loading. Took %llu ms\n\n", duration.count());

    printf("Started MNIST net building...\n");
    start = high_resolution_clock::now();
    nn::NeuralNet netMNIST({ 784, 5, 5, 5, 10 });
    stop = high_resolution_clock::now();
    duration = duration_cast<milliseconds>(stop - start);
    printf("Ended MNIST net building. Took %llu ms\n\n", duration.count());

    printf("Started MNIST net training...\n");
    start = high_resolution_clock::now();
    dataset::train(netMNIST, mnist, 100, false);
    stop = high_resolution_clock::now();
    duration = duration_cast<milliseconds>(stop - start);
    printf("Ended MNIST net training. Took %llu ms\n\n", duration.count());
    netMNIST.save("/Users/dankulakovich/CLionProjects/nn/models/MNIST.model");
    return 0;
}
