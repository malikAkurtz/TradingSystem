#include "NetworkLayers.h"
#include "Neuron.h"
#include "Output.h"
#include <iostream>

std::vector<double> NUM_NEURONS_PER_LAYER = {};

// InputLayer Constructor
InputLayer::InputLayer(int num_features) {
    for (int i = 0; i < num_features; ++i) {
        InputNeuron neuron;
        this->inputNeurons.push_back(neuron);
    }
    NUM_NEURONS_PER_LAYER.push_back(num_features);
}

// InputLayer::calculateLayerOutputs
std::vector<std::vector<double>> InputLayer::getInputs() {
    return this->inputs;
}
void InputLayer::storeInputs(std::vector<std::vector<double>> input_vector) {
    this->inputs = input_vector;
}

void InputLayer::addNeuron(InputNeuron inputNeuron) {
    inputNeurons.push_back(inputNeuron);
}



////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////




// Layer Constructor
Layer::Layer(int num_neurons, ActivationFunction type)
    : AFtype(type) {
    for (int i = 0; i < num_neurons; ++i) {
        printDebug("Number of neuron in previous layer");
        print(NUM_NEURONS_PER_LAYER.back());
        Neuron neuron(NUM_NEURONS_PER_LAYER.back() + 1); // Including bias
        this->addNeuron(neuron);
        this->addNeuronWeights(neuron);
    }
    NUM_NEURONS_PER_LAYER.push_back(num_neurons);
}

void Layer::calculateLayerOutputs(std::vector<std::vector<double>> input_vector) {
    input_vector.push_back({1}); // Add bias

    printDebug("Weights Matrix for this layer");
    printMatrixDebug(this->getWeightsMatrix());
    this->pre_activation_outputs = (matrixMultiply(this->getWeightsMatrix(), input_vector));
    if (AFtype == RELU) {
        this->activation_outputs = ReLU(this->pre_activation_outputs);
        this->derivative_activation_outputs = d_ReLU(this->pre_activation_outputs);
        
    } else if(AFtype == SIGMOID) {
        this->activation_outputs = sigmoid(this->pre_activation_outputs);
        this->derivative_activation_outputs = d_sigmoid(this->pre_activation_outputs);
    } else {
        this->activation_outputs = this->pre_activation_outputs;
        this->derivative_activation_outputs = this->pre_activation_outputs;
    }
}

std::vector<std::vector<double>> Layer::getActivationOutputs() {
    return this->activation_outputs;
}

std::vector<std::vector<double>> Layer::getPreActivationOutputs() {
    return this->pre_activation_outputs;
}

std::vector<std::vector<double>> Layer::getDerivativeActivationOutputs() {
    return this->derivative_activation_outputs;
}

std::vector<std::vector<double>> Layer::getWeightsMatrix() {
    return this->weightsMatrix;
}

void Layer::addNeuron(Neuron neuron) {
    this->neurons.push_back(neuron);
}

void Layer::addNeuronWeights(Neuron neuron) {
        this->weightsMatrix.push_back(neuron.getWeights());
    }

void Layer::updateNeuronWeights(std::vector<std::vector<double>> gradient_matrix, float LR) {
        for (int i = 0; i < gradient_matrix.size(); i++) {
            for (int j = 0; j < gradient_matrix[i].size(); j++) {
                this->weightsMatrix[i][j] -= (LR * gradient_matrix[i][j]);
                this->neurons[i].weights[j] -=(LR * gradient_matrix[i][j]);
            }
        }
    }




////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
