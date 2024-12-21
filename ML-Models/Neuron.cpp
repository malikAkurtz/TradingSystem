#include "Neuron.h"
#include "Output.h"

// InputNeuron Definitions
InputNeuron::InputNeuron() {}



///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////




// Neuron Definitions
Neuron::Neuron(int num_weights) {
    if (DEBUG) {
        weights.resize(num_weights, static_cast<double>(rand()) / RAND_MAX - 0.5);
    } else {
        weights.resize(num_weights, 2);
    }
}

std::vector<double> Neuron::getWeights() const {
    return weights;
}

void Neuron::setWeights(const std::vector<double>& new_weights) {
    weights = new_weights;
}
