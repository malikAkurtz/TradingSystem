#include "NetworkLayers.h"
#include "Neuron.h"
#include "Output.h"

// InputLayer Constructor
InputLayer::InputLayer(int num_features) {
    for (int i = 0; i < num_features; ++i) {
        InputNeuron neuron;
        inputNeurons.push_back(neuron);
    }
}

// InputLayer::calculateLayerOutputs
void InputLayer::calculateLayerOutputs(std::vector<double> input_vector) {
    pre_activation_outputs = input_vector;
}

std::vector<double> InputLayer::getPreActivationOutputs() {
    return pre_activation_outputs;
}

void InputLayer::addNeuron(InputNeuron inputNeuron) {
    inputNeurons.push_back(inputNeuron);
}



////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////




// HiddenLayer Constructor
HiddenLayer::HiddenLayer(int num_neurons, int num_neurons_in_prev_layer, ActivationFunction type)
    : AFtype(type) {
    for (int i = 0; i < num_neurons; ++i) {
        Neuron neuron(num_neurons_in_prev_layer + 1); // Including bias
        addNeuron(neuron);
        addNeuronWeights(neuron);
    }
}

// HiddenLayer::calculateLayerOutputs
void HiddenLayer::calculateLayerOutputs(std::vector<double> input_vector) {
    input_vector.push_back(1); // Add bias

    pre_activation_outputs = matrixToVector(
        matrixMultiply(getWeightsMatrix(), vectorToMatrix(input_vector)));
    applyActivation();
}

std::vector<double> HiddenLayer::getActivationOutputs() {
    return activation_outputs;
}

std::vector<double> HiddenLayer::getPreActivationOutputs() {
    return pre_activation_outputs;
}

// HiddenLayer::applyActivation
void HiddenLayer::applyActivation() {
    activation_outputs.resize(pre_activation_outputs.size());
    if (AFtype == RELU) {
        for (size_t i = 0; i < pre_activation_outputs.size(); ++i) {
            activation_outputs[i] = ReLU(pre_activation_outputs[i]);
        }
    } else if (AFtype == SIGMOID) {
        for (size_t i = 0; i < pre_activation_outputs.size(); ++i) {
            activation_outputs[i] = sigmoid(pre_activation_outputs[i]);
        }
    }
}

std::vector<std::vector<double>> HiddenLayer::getWeightsMatrix() {
    return weightsMatrix;
}

void HiddenLayer::addNeuron(Neuron neuron) {
    neurons.push_back(neuron);
}

void HiddenLayer::addNeuronWeights(Neuron neuron) {
        weightsMatrix.push_back(neuron.getWeights());
    }




////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////




// OutputLayer Constructor
OutputLayer::OutputLayer(int num_neurons, int num_neurons_in_prev_layer, ActivationFunction type)
    : AFtype(type) {
    for (int i = 0; i < num_neurons; ++i) {
        Neuron neuron(num_neurons_in_prev_layer + 1); // Including bias
        addNeuron(neuron);
        addNeuronWeights(neuron);
    }
}

// OutputLayer::calculateLayerOutputs
void OutputLayer::calculateLayerOutputs(std::vector<double> input_vector) {
    input_vector.push_back(1); // Add bias
    // printVector(input_vector);
    // printMatrix(getWeightsMatrix());
    pre_activation_outputs = matrixToVector(
        matrixMultiply(weightsMatrix, vectorToMatrix(input_vector)));
    // printVector(pre_activation_outputs);
    applyActivation();
}

std::vector<double> OutputLayer::getActivationOutputs() {
    return activation_outputs;
}

std::vector<double> OutputLayer::getPreActivationOutputs() {
    return pre_activation_outputs;
}

// OutputLayer::applyActivation
void OutputLayer::applyActivation() {
    activation_outputs.resize(pre_activation_outputs.size());
    if (AFtype == RELU) {
        for (size_t i = 0; i < pre_activation_outputs.size(); ++i) {
            activation_outputs[i] = ReLU(pre_activation_outputs[i]);
        }
    } else if (AFtype == SIGMOID) {
        for (size_t i = 0; i < pre_activation_outputs.size(); ++i) {
            activation_outputs[i] = sigmoid(pre_activation_outputs[i]);
        }
    } else {
        activation_outputs = pre_activation_outputs;
    }
}

std::vector<std::vector<double>> OutputLayer::getWeightsMatrix() const{
    return weightsMatrix;
}

void OutputLayer::addNeuron(Neuron neuron) {
    neurons.push_back(neuron);
}

void OutputLayer::addNeuronWeights(Neuron neuron) {
        weightsMatrix.push_back(neuron.getWeights());
    }
