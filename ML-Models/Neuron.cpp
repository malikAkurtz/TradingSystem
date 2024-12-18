#include "Neuron.h"

// InputNeuron Definitions
InputNeuron::InputNeuron() : input(0.0) {}

double InputNeuron::getInput() const {
    return input;
}

void InputNeuron::setInput(double new_input) {
    input = new_input;
}



///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////




// Neuron Definitions
Neuron::Neuron(int num_weights) {
    weights.resize(num_weights, 2.0);
}

std::vector<double> Neuron::getWeights() const {
    return weights;
}

void Neuron::setWeights(const std::vector<double>& new_weights) {
    weights = new_weights;
}
