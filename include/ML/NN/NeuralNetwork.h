#ifndef NEURALNETWORK_H
#define NEURALNETWORK_H

#include <vector>
#include "NetworkLayers.h"
#include "LossFunctions.h"
#include "LinearAlgebra.h"
#include "LossFunctionTypes.h"

class Optimizer;
class NeuralNetwork
{
public:
    // Member variables
    Optimizer *optimizer;
    int num_hidden_layers;
    InputLayer input_layer;
    std::vector<Layer> layers;
    double model_loss;
    std::vector<int> layer_sizes;

    // Constructor
    NeuralNetwork();
    NeuralNetwork(Optimizer* optimizer);
    NeuralNetwork(const NeuralNetwork &base_NN, const std::vector<double> &encoding);

    // Public methods
    double calculateFinalModelLoss(std::vector<std::vector<double>> features_matrix, std::vector<std::vector<double>> labels);
    void fit(const std::vector<std::vector<double>>& features_matrix, const std::vector<std::vector<double>>& labels);
    std::vector<std::vector<double>> feedForward(std::vector<std::vector<double>> features_matrix);
    void addInputLayer(int num_features);
    void addLayer(int num_neurons, ActivationFunctionType activation_function, NeuronInitializationType neuron_initialization);
    void reInitializeLayers();
    std::vector<double> getNetworkEncoding() const;
    void setEncoding(std::vector<double> encoding);
};

#endif // NEURALNETWORK_H
