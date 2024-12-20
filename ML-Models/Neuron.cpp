#include "Neuron.h"

// InputNeuron Definitions
InputNeuron::InputNeuron() {}



///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////




// Neuron Definitions
Neuron::Neuron(int num_weights) {
    weights.resize(num_weights, 2); //
}

std::vector<double> Neuron::getWeights() const {
    return weights;
}

void Neuron::setWeights(const std::vector<double>& new_weights) {
    weights = new_weights;
}
