#ifndef NETWORKLAYERS_H
#define NETWORKLAYERS_H

#include <vector>
#include <memory>
#include "Node.h"
#include "LinearAlgebra.h"
#include "GenFunctions.h"
#include "/Users/malikkurtz/Coding/TradingSystem/include/libs/Math/ActivationFunctions.h"
#include "../NodeInitializationType.h"
#include "../ActivationFunctionTypes.h"


////////////////////////////////////////////////////////////////////////////////////////////////////


// Input Layer Class
class InputLayer {
public:
    std::vector<InputNode> input_nodes;
    std::vector<std::vector<double>> inputs;

    explicit InputLayer();
    // Constructor
    explicit InputLayer(int num_features);

    // Calculate outputs for the input layer
    std::vector<std::vector<double>> getInputs();

    void storeInputs(std::vector<std::vector<double>> input_vector);


    void addNode(InputNode node);
};


////////////////////////////////////////////////////////////////////////////////////////////////////


// Layer Class
class Layer {
    public:
    std::vector<Node> nodes;
    ActivationFunctionType activation_function;
    NodeInitializationType node_initalization;
    std::vector<std::vector<double>> pre_activation_outputs;
    std::vector<std::vector<double>> activation_outputs;
    std::vector<std::vector<double>> derivative_activation_outputs;
    

    // Constructor
    Layer(int num_nodes, int num_inputs, ActivationFunctionType activation_function, NodeInitializationType node_initalization);

    // Calculate outputs for the hidden layer
    void calculateLayerOutputs(std::vector<std::vector<double>> input_vector);

    // Getter for activation outputs
    std::vector<std::vector<double>> getActivationOutputs();

    // Getter for pre-activation outputs
    std::vector<std::vector<double>> getPreActivationOutputs();

    std::vector<std::vector<double>> getDerivativeActivationOutputs();

    // Getter for weights matrix
    std::vector<std::vector<double>> getWeightsMatrix();

    void addNode(Node node);

    void updateNodeWeights(std::vector<std::vector<double>> gradient_matrix, float learning_rate);

    void reInitializeNodes();
};

////////////////////////////////////////////////////////////////////////////////////////////////////

#endif