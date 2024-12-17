#include <vector>
#include "LinearAlgebra.h"
#include "GenFunctions.h"

enum ActivationFunction {
    RELU,
    SIGMOID,
};

class Neuron {
    public:
    std::vector<double> parameters;
    std::vector<double> input;
    double output;


    Neuron() {
    }

    Neuron(std::vector<double> input) {
        this->input = input;
    }
 
    double f(std::vector<double> input_vector, std::vector<double> parameters) {
        std::vector<double> ones_added = input_vector;
        addElement(ones_added, 1, 0);
        this->output = innerProduct(ones_added, parameters);

        return this->output;
    }

    // double calculateLoss(std::vector<double> input_vector, std::vector<double> parameters, std::vector<double> labels) {
    //     std::vector<double> predictions = f(featuresMatrix, parameters);
    //     double MSE = calculateMSE(predictions, labels);
    //     return MSE;
    // }
};


class NetworkLayer {
    public:
    std::vector<Neuron> nodes;
    ActivationFunction AFtype = RELU;

    NetworkLayer(ActivationFunction type) {
        this->AFtype = type;
    }

    NetworkLayer() {
    }

    // probably need to change this to take in a vector
    double applyActivation(double value) {
        if (this->AFtype == RELU) {
            return ReLU(value);
        } else if (this->AFtype == SIGMOID) {
            return sigmoid(value);
        }
    }

    void feedForward(std::vector<double> to_foward, NetworkLayer& next_layer) {

    }
};

class NeuralNetwork {
    public:
    std::vector<NetworkLayer> layers;

    NeuralNetwork() {
    }
};