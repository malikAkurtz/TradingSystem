#include "Neuron.h"


// InputNeuron Definitions
InputNeuron::InputNeuron() {}



///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////


// Neuron Definitions
Neuron::Neuron(int num_weights, NeuronInitialization type) {
    if (type == RANDOM) {
        weights.resize(num_weights, static_cast<double>(rand()) / RAND_MAX - 0.5); // 
    }
    else if (type == CONSTANT) {
        weights.resize(num_weights, 2);
    } else {
        throw std::invalid_argument("Neuron Initialization not specified!");
    }
}

std::vector<double> Neuron::getWeights() const {
    return weights;
}

void Neuron::setWeights(const std::vector<double>& new_weights) {
    weights = new_weights;
}

void Neuron::reInitializeWeights(NeuronInitialization type)
{
    if (type == RANDOM) {
        weights.resize(weights.size(), static_cast<double>(rand()) / RAND_MAX - 0.5); // 
    }
    else if (type == CONSTANT) {
        weights.resize(weights.size(), 2);
    } else {
        throw std::invalid_argument("Neuron Initialization not specified!");
    }
}