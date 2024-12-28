#ifndef NEURON_H
#define NEURON_H

#include <vector>
#include <random>
#include "NeuronInitializationType.h"
#include "NodeType.h"
#include "ActivationFunctionTypes.h"

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
    // For GD and NE
    std::vector<double> weights;

    // Constructor to initialize weights to a default value
    explicit Neuron(int num_weights, NeuronInitializationType initialization_type);


    void reInitializeWeights(NeuronInitializationType initialization_type);
};


#endif // NEURON_H
