#ifndef NETWORKLAYERS_H
#define NETWORKLAYERS_H

#include <vector>
#include <memory>
#include "Neuron.h"
#include "LinearAlgebra.h"
#include "GenFunctions.h"

// Activation Function Enum
enum ActivationFunction {
    RELU,
    SIGMOID,
    NONE
};

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////


// Input Layer Class
class InputLayer {
public:
    std::vector<InputNeuron> inputNeurons;
    std::vector<std::vector<double>> inputs;

    // Constructor
    explicit InputLayer(int num_features);

    // Calculate outputs for the input layer
    std::vector<std::vector<double>> getInputs();

    void storeInputs(std::vector<std::vector<double>> input_vector);


    void addNeuron(InputNeuron neuron);
};



////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////



// Layer Class
class Layer {
    public:
    std::vector<Neuron> neurons;
    ActivationFunction AFtype;
    std::vector<std::vector<double>> weightsMatrix;
    std::vector<std::vector<double>> pre_activation_outputs;
    std::vector<std::vector<double>> activation_outputs;
    std::vector<std::vector<double>> derivative_activation_outputs;
    

    // Constructor
    Layer(int num_neurons, int num_neurons_in_prev_layer, ActivationFunction type);

    // Calculate outputs for the hidden layer
    void calculateLayerOutputs(std::vector<std::vector<double>> input_vector);

    // Getter for activation outputs
    std::vector<std::vector<double>> getActivationOutputs();

    // Getter for pre-activation outputs
    std::vector<std::vector<double>> getPreActivationOutputs();

    std::vector<std::vector<double>> getDerivativeActivationOutputs();

    // Getter for weights matrix
    std::vector<std::vector<double>> getWeightsMatrix();

    void addNeuron(Neuron neuron);

    void addNeuronWeights(Neuron neuron);

    void updateNeuronWeights(std::vector<std::vector<double>> gradient_matrix, float LR);
};



////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

#endif