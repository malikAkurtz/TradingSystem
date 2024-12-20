#include "NetworkLayers.h"
#include "Neuron.h"
#include "Output.h"
#include <iostream>

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

    // print("input vector");
    // printVector(input_vector);
    // print("weights matrix");
    // printMatrix(getWeightsMatrix());
    pre_activation_outputs = columnVectortoVector1D(
        matrixMultiply(getWeightsMatrix(), vector1DtoColumnVector(input_vector)));

    // std::cout << "pre_activation_outputs" << std::endl;
    // printVector(pre_activation_outputs);
    if (AFtype == RELU) {
        this->activation_outputs = ReLU(this->pre_activation_outputs);
        this->derivative_activation_outputs = d_ReLU(this->pre_activation_outputs);
    } else if(AFtype == SIGMOID) {
        throw std::invalid_argument("Derivative for SIGMOID not implemented");
        this->activation_outputs = sigmoid(this->pre_activation_outputs);
    } else {
        this->activation_outputs = this->pre_activation_outputs;
        this->derivative_activation_outputs = this->pre_activation_outputs;
    }
}

std::vector<double> HiddenLayer::getActivationOutputs() {
    return activation_outputs;
}

std::vector<double> HiddenLayer::getPreActivationOutputs() {
    return pre_activation_outputs;
}

std::vector<double> HiddenLayer::getDerivativeActivationOutputs() {
    return derivative_activation_outputs;
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

void HiddenLayer::updateNeuronWeights(std::vector<std::vector<double>> gradient_matrix, float LR) {
        // for every gradient in the gradient matrix
        for (int i = 0; i < gradient_matrix.size(); i++) {
            for (int j = 0; j < gradient_matrix[i].size(); j++) {
                // std::cout << "For i, j: " << i << ", " << j << std::endl;
                // std::cout << "gradient_matrix[i][j] -> " << gradient_matrix[i][j] << std::endl;
                this->weightsMatrix[i][j] -= (LR * gradient_matrix[i][j]);
                this->neurons[i].weights[j] -=(LR * gradient_matrix[i][j]);
            }
        }
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

    // print("Input Vector");
    // printVector(input_vector);
    // printVectorShape(input_vector);
    
    // print("getWeightsMatrix()");
    // printMatrix(this->getWeightsMatrix());
    // printMatrixShape(this->getWeightsMatrix());

    // print("vector1Dto2D(input_vector)");
    // printMatrix(vector1Dto2D(input_vector));
    // printMatrixShape(vector1Dto2D(input_vector));

    this->pre_activation_outputs = columnVectortoVector1D(
        matrixMultiply(this->getWeightsMatrix(), vector1DtoColumnVector(input_vector)));
    if (AFtype == RELU) {
        this->activation_outputs = ReLU(this->pre_activation_outputs);
        this->derivative_activation_outputs = d_ReLU(this->pre_activation_outputs);
    } else if(AFtype == SIGMOID) {
        throw std::invalid_argument("Derivative for SIGMOID not implemented");
        this->activation_outputs = sigmoid(this->pre_activation_outputs);
    } else {
        this->activation_outputs = this->pre_activation_outputs;
        this->derivative_activation_outputs = this->pre_activation_outputs;
    }
}

std::vector<double> OutputLayer::getActivationOutputs() {
    return activation_outputs;
}

std::vector<double> OutputLayer::getPreActivationOutputs() {
    return pre_activation_outputs;
}

std::vector<double> OutputLayer::getDerivativeActivationOutputs() {
    return derivative_activation_outputs;
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

void OutputLayer::updateNeuronWeights(std::vector<std::vector<double>> gradient_matrix, float LR) {
        // for every gradient in the gradient matrix
        for (int i = 0; i < gradient_matrix.size(); i++) {
            for (int j = 0; j < gradient_matrix[0].size(); j++) {
                this->weightsMatrix[i][j] -= (LR * gradient_matrix[i][j]);
                this->neurons[i].weights[j] -=(LR * gradient_matrix[i][j]);
            }
        }
    }