#include "NetworkLayers.h"


using namespace LinearAlgebra;
using namespace ActivationFunctions;

// InputLayer Constructor
InputLayer::InputLayer() {}

InputLayer::InputLayer(int num_features) {
    for (int i = 0; i < num_features; ++i) {
        InputNode node;
        this->input_nodes.push_back(node);
    }
}

// InputLayer::calculateLayerOutputs
std::vector<std::vector<double>> InputLayer::getInputs() {
    return this->inputs;
}
void InputLayer::storeInputs(std::vector<std::vector<double>> input_matrix) {
    this->inputs = input_matrix;
}

void InputLayer::addNode(InputNode input_node) {
    this->input_nodes.push_back(input_node);
}



////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////



// Layer Constructor
Layer::Layer(int num_nodes, int num_inputs, ActivationFunctionType activation_function, NodeInitializationType node_initialization) : activation_function(activation_function), node_initalization(node_initialization) 
{
    for (int i = 0; i < num_nodes; ++i) {
        Node node(num_inputs + 1, node_initialization); // Including bias
        this->addNode(node);
    }
}

void Layer::calculateLayerOutputs(std::vector<std::vector<double>> input_matrix) 
{
    int num_columns = input_matrix[0].size();
    std::vector<double> ones_to_append(num_columns, 1);

    input_matrix.push_back(ones_to_append); // Add bias
    // printDebug("Weights lool like");
    // printMatrixDebug(this->getWeightsMatrix());
    this->pre_activation_outputs = (matrixMultiply(this->getWeightsMatrix(), input_matrix));
    if (activation_function == RELU) 
    {
        this->activation_outputs = matrix_ReLU(this->pre_activation_outputs);
        this->derivative_activation_outputs = matrix_d_ReLU(this->pre_activation_outputs);
    } 
    else if (activation_function == SIGMOID) 
    {
        this->activation_outputs = matrix_sigmoid(this->pre_activation_outputs);
        this->derivative_activation_outputs = matrix_d_sigmoid(this->pre_activation_outputs);
    } 
    else if (activation_function == NONE) 
    {
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
    std::vector<std::vector<double>> weights_matrix;
    for (int i = 0; i < nodes.size(); i++) {
        // printDebug("This Neurons Weights");
        // std::vector<double> theweights = neurons[i].getWeights();
        // for (int m = 0; m < theweights.size(); m++)
        // {
        //     printDebug(theweights[i]);
        // }
        weights_matrix.push_back(nodes[i].weights);
    }

    // print("SO weights matrix is");
    // printMatrixDebug(weightsMatrix);
    return weights_matrix;
}

void Layer::addNode(Node node) {
    this->nodes.push_back(node);
}

void Layer::updateNodeWeights(std::vector<std::vector<double>> gradient_matrix, float LR) 
{
    for (int i = 0; i < gradient_matrix.size(); i++) {
        for (int j = 0; j < gradient_matrix[i].size(); j++) {
            this->nodes[i].weights[j] -=(LR * gradient_matrix[i][j]);
        }
    }
}

void Layer::reInitializeNodes()
{
    for (Node node : nodes)
    {
        node.reInitializeWeights(this->node_initalization);
    }
}

////////////////////////////////////////////////////////////////////////////////////////////////////
