#include "Neuron.h"


// InputNeuron Definitions
InputNeuron::InputNeuron() {}


////////////////////////////////////////////////////////////////////////////////////////////////////


// Neuron Definitions
Neuron::Neuron(int num_weights, NeuronInitializationType initialization_type) 
{
    if (initialization_type == RANDOM) 
    {
        weights.resize(num_weights, static_cast<double>(rand()) / RAND_MAX - 0.5); // 
    }
    else if (initialization_type == CONSTANT) 
    {
        weights.resize(num_weights, 2);
    } 
    else 
    {
        throw std::invalid_argument("Neuron Initialization not specified!");
    }
}

void Neuron::reInitializeWeights(NeuronInitializationType initialization_type)
{
    if (initialization_type == RANDOM) 
    {
        weights.resize(weights.size(), static_cast<double>(rand()) / RAND_MAX - 0.5); // 
    }
    else if (initialization_type == CONSTANT) 
    {
        weights.resize(weights.size(), 2);
    } 
    else 
    {
        throw std::invalid_argument("Neuron Initialization not specified!");
    }
}