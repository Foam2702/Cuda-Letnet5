#include <Eigen/Dense>
#include <algorithm>
#include <iostream>

#include "src/layer.h"
#include "src/layer/conv.h"
#include "src/layer/conv_gpu.h"
#include "src/layer/fully_connected.h"
#include "src/layer/ave_pooling.h"
#include "src/layer/max_pooling.h"
#include "src/layer/relu.h"
#include "src/layer/sigmoid.h"
#include "src/layer/softmax.h"
#include "src/loss.h"
#include "src/loss/mse_loss.h"
#include "src/loss/cross_entropy_loss.h"
#include "src/mnist.h"
#include "src/network.h"
#include "src/optimizer.h"
#include "src/optimizer/sgd.h"
#include "src/layer/parallel/gpu_support.h"
#include "dnnNetwork.h"

int main()
{
    GPU_Support gpu_support;
    MNIST dataset("./data/mnist/");
    Network dnn1 = dnnNetwork_CPU();
    Network dnn2 = dnnNetwork_GPU();

    dataset.read();
    int n_train = dataset.train_data.cols();
    int dim_in = dataset.train_data.rows();
    std::cout << "mnist train number: " << n_train << std::endl;
    std::cout << "mnist test number: " << dataset.test_labels.cols() << std::endl;
    float accuracy = 0.0;

    std::cout << "CPU" << std::endl;
    dnn1.load_parameters("./model/weight_model_cpu.bin");
    std::cout << "\nOKE\n";
    dnn1.forward(dataset.test_data);
    accuracy = compute_accuracy(dnn1.output(), dataset.test_labels);
    std::cout << "test accuracy: " << accuracy << std::endl;

    std::cout << "GPU" << std::endl;
    dnn2.load_parameters("./model/weight_model.bin");
    std::cout << "\nOKE\n";
    dnn2.forward(dataset.test_data);
    accuracy = compute_accuracy(dnn2.output(), dataset.test_labels);
    std::cout << "test accuracy: " << accuracy << std::endl;
}