#ifndef NETWORKLAYERS_H
#define NETWORKLAYERS_H

#include <vector>
#include <memory>
#include "Neuron.h"
#include "LinearAlgebra.h"
#include "GenFunctions.h"
#include "ActivationFunctions.h"
#include "NeuronInitializationType.h"
#include "ActivationFunctionTypes.h"


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
    ActivationFunctionType AFtype;
    NeuronInitializationType initalization;
    std::vector<std::vector<double>> pre_activation_outputs;
    std::vector<std::vector<double>> activation_outputs;
    std::vector<std::vector<double>> derivative_activation_outputs;
    

    // Constructor
    Layer(int num_neurons, int num_inputs, ActivationFunctionType AFtype, NeuronInitializationType NItype);

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

    void updateNeuronWeights(std::vector<std::vector<double>> gradient_matrix, float LR);

    void reInitializeNeurons();
};

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

#endif