#ifndef NEURON_H
#define NEURON_H

#include <vector>
#include <random>
#include "NeuronInitializationType.h"

// InputNeuron Class
class InputNeuron {
public:
    // Default constructor
    InputNeuron();
};

////////////////////////////////////////////////////////////////////////////////////////////////////

// Neuron Class
class Neuron {
public:
    // Member variable
    std::vector<double> weights;

    // Constructor to initialize weights to a default value
    explicit Neuron(int num_weights, NeuronInitializationType initialization_type);

    // Getter for weights
    std::vector<double> getWeights() const;

    // Setter for weights
    void setWeights(const std::vector<double>& new_weights);

    void reInitializeWeights(NeuronInitializationType initialization_type);
};


#endif // NEURON_H
