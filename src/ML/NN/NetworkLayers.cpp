#include "NetworkLayers.h"


using namespace LinearAlgebra;
using namespace ActivationFunctions;

// InputLayer Constructor
InputLayer::InputLayer() {}

InputLayer::InputLayer(int num_features) {
    for (int i = 0; i < num_features; ++i) {
        InputNeuron neuron;
        this->inputNeurons.push_back(neuron);
    }
}

// InputLayer::calculateLayerOutputs
std::vector<std::vector<double>> InputLayer::getInputs() {
    return this->inputs;
}
void InputLayer::storeInputs(std::vector<std::vector<double>> input_matrix) {
    this->inputs = input_matrix;
}

void InputLayer::addNeuron(InputNeuron inputNeuron) {
    inputNeurons.push_back(inputNeuron);
}



////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////



// Layer Constructor
Layer::Layer(int num_neurons, int num_inputs, ActivationFunctionType AFtype, NeuronInitializationType NItype)
    : AFtype(AFtype), initalization(NItype) {
    for (int i = 0; i < num_neurons; ++i) {
        Neuron neuron(num_inputs + 1, NItype); // Including bias
        this->addNeuron(neuron);
    }
}

void Layer::calculateLayerOutputs(std::vector<std::vector<double>> input_matrix) {
    int num_columns = input_matrix[0].size();
    std::vector<double> ones_to_append(num_columns, 1);

    input_matrix.push_back(ones_to_append); // Add bias
    // printDebug("Weights lool like");
    // printMatrixDebug(this->getWeightsMatrix());
    this->pre_activation_outputs = (matrixMultiply(this->getWeightsMatrix(), input_matrix));
    if (AFtype == RELU) {
        this->activation_outputs = matrix_ReLU(this->pre_activation_outputs);
        this->derivative_activation_outputs = matrix_d_ReLU(this->pre_activation_outputs);
        } else if(AFtype == SIGMOID) {
        this->activation_outputs = matrix_sigmoid(this->pre_activation_outputs);
        this->derivative_activation_outputs = matrix_d_sigmoid(this->pre_activation_outputs);
    } else if (AFtype == NONE) {
        this->activation_outputs = this->pre_activation_outputs;
        this->derivative_activation_outputs = createOnesMatrix(this->pre_activation_outputs.size(), this->pre_activation_outputs[0].size());
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
    // printDebug("Number of Neurons in layer");
    // printDebug(neurons.size());
    std::vector<std::vector<double>> weightsMatrix;
    for (int i = 0; i < neurons.size(); i++) {
        // printDebug("This Neurons Weights");
        // std::vector<double> theweights = neurons[i].getWeights();
        // for (int m = 0; m < theweights.size(); m++)
        // {
        //     printDebug(theweights[i]);
        // }
        weightsMatrix.push_back(neurons[i].getWeights());
    }

    // print("SO weights matrix is");
    // printMatrixDebug(weightsMatrix);
    return weightsMatrix;
}

void Layer::addNeuron(Neuron neuron) {
    this->neurons.push_back(neuron);
}

void Layer::updateNeuronWeights(std::vector<std::vector<double>> gradient_matrix, float LR) {
        for (int i = 0; i < gradient_matrix.size(); i++) {
            for (int j = 0; j < gradient_matrix[i].size(); j++) {
                this->neurons[i].weights[j] -=(LR * gradient_matrix[i][j]);
            }
        }
}

void Layer::reInitializeNeurons()
{
    for (Neuron neuron : neurons)
    {
        neuron.reInitializeWeights(this->initalization);
    }
}

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
