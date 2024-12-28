#ifndef NODE_H
#define NODE_H

#include <vector>
#include <random>
#include "NodeInitializationType.h"
#include "NodeType.h"
#include "ActivationFunctionTypes.h"

// InputNeuron Class
class InputNode {
public:
    // Default constructor
    InputNode();
};

////////////////////////////////////////////////////////////////////////////////////////////////////

// Neuron Class
class Node {
public:
    
    std::vector<double> weights;

    // Constructor to initialize weights to a default value
    explicit Node(int num_weights, NodeInitializationType initialization_type);


    void reInitializeWeights(NodeInitializationType initialization_type);
};


#endif
