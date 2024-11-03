#include <chrono>
#include <cstdio>
#include "neural_net/nnet.h"
#include "dataset/dataset.h"

using namespace std::chrono;

int main() {


//    -------------TRAIN MODEL-------------

    // nn::NeuralNet net({ 2, 3, 3, 1 });
    //
    // dataset::FileDataset trainset("/home/dan/CLionProjects/neunet/train/XOR/train.dset");
    // dataset::train(net, trainset, 1000000, false, 0.00001);
    //
    // net.save("/home/dan/CLionProjects/neunet/models/XOR.model");

//    -------------LOAD AND TEST-------------

    // nn::NeuralNet net1;
    // dataset::FileDataset testset("/home/dan/CLionProjects/neunet/test/XOR/test.dset");
    // net1.load("/home/dan/CLionProjects/neunet/models/XOR.model");
    // dataset::test(net1, testset, true, 0.01);

//    -------------MNIST DATASET-------------
    // printf("Started MNIST dataset loading\n");
    // auto start = high_resolution_clock::now();
    // const char* train_images_path = "/home/dan/CLionProjects/neunet/train/MNIST/train-images.idx3-ubyte";
    // const char* train_labels_path = "/home/dan/CLionProjects/neunet/train/MNIST/train-labels.idx1-ubyte";
    // dataset::MNISTDataset mnist(train_images_path, train_labels_path, 6000);
    // auto stop = high_resolution_clock::now();
    // auto duration = duration_cast<milliseconds>(stop - start);
    // printf("Ended MNIST dataset loading. Took %llu ms\n\n", duration.count());
    //
    // printf("Started MNIST net building...\n");
    // start = high_resolution_clock::now();
    // NeuralNet netMNIST({ 784, 20, 10, 10 });
    // stop = high_resolution_clock::now();
    // duration = duration_cast<milliseconds>(stop - start);
    // printf("Ended MNIST net building. Took %llu ms\n\n", duration.count());
    //
    // printf("Started MNIST net training...\n");
    // start = high_resolution_clock::now();
    // train(netMNIST, mnist, 100, false);
    // stop = high_resolution_clock::now();
    // duration = duration_cast<milliseconds>(stop - start);
    // printf("Ended MNIST net training. Took %llu ms\n\n", duration.count());
    // netMNIST.save("/home/dan/CLionProjects/neunet/models/MNIST.model");

    nn::NeuralNet mnistTest;
    dataset::MNISTDataset testset("/home/dan/CLionProjects/neunet/train/MNIST/train-images.idx3-ubyte", "/home/dan/CLionProjects/neunet/train/MNIST/train-labels.idx1-ubyte", 10000);
    mnistTest.load("/home/dan/CLionProjects/neunet/models/MNIST.model");
    dataset::test(mnistTest, testset, true, 0.01);

    return 0;
}
