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

// Base Layer Class
class Layer {
public:
    std::vector<Neuron> neurons;
    std::vector<std::vector<double>> pre_activation_outputs;

    virtual ~Layer() {} // Virtual destructor for polymorphism

    // Pure virtual function for calculating layer outputs
    virtual void calculateLayerOutputs(std::vector<std::vector<double>> input_vector) = 0;
        
};



////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////




// Input Layer Class
class InputLayer : public Layer {
public:
    std::vector<InputNeuron> inputNeurons;

    // Constructor
    explicit InputLayer(int num_features);

    // Calculate outputs for the input layer
    void calculateLayerOutputs(std::vector<std::vector<double>> input_vector) override;

    // Getter for pre-activation outputs
    std::vector<std::vector<double>> getPreActivationOutputs();

    void addNeuron(InputNeuron neuron);
};



////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////




// Hidden Layer Class
class HiddenLayer : public Layer {
public:
    ActivationFunction AFtype;
    std::vector<std::vector<double>> activation_outputs;
    std::vector<std::vector<double>> derivative_activation_outputs;
    std::vector<std::vector<double>> weightsMatrix;

    // Constructor
    HiddenLayer(int num_neurons, int num_neurons_in_prev_layer, ActivationFunction type);

    // Calculate outputs for the hidden layer
    void calculateLayerOutputs(std::vector<std::vector<double>> input_vector) override;

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




// Output Layer Class
class OutputLayer : public Layer {
public:
    ActivationFunction AFtype;
    std::vector<std::vector<double>> activation_outputs;
    std::vector<std::vector<double>> derivative_activation_outputs;
    std::vector<std::vector<double>> weightsMatrix;

    // Constructor
    OutputLayer(int num_neurons, int num_neurons_in_prev_layer, ActivationFunction type);

    // Calculate outputs for the output layer
    void calculateLayerOutputs(std::vector<std::vector<double>> input_vector) override;

    // Getter for activation outputs
    std::vector<std::vector<double>> getActivationOutputs();

    // Getter for pre-activation outputs
    std::vector<std::vector<double>> getPreActivationOutputs();

    std::vector<std::vector<double>> getDerivativeActivationOutputs();

    // Apply activation function to the layer's outputs
    void applyActivation();

    // Getter for weights matrix
    std::vector<std::vector<double>> getWeightsMatrix() const;

    void addNeuron(Neuron neuron);

    void addNeuronWeights(Neuron neuron);

    void updateNeuronWeights(std::vector<std::vector<double>> gradient_matrix, float LR);
};

#endif // NETWORKLAYERS_H
