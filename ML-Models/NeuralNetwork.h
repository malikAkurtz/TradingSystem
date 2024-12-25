#ifndef NEURALNETWORK_H
#define NEURALNETWORK_H

#include <vector>
#include "NetworkLayers.h"
#include "LossFunctions.h"
#include "LinearAlgebra.h"
#include "OptimizationMethods.h"
#include "OptimizationTypes.h"




class NeuralNetwork {
public:
    // Member variables
    float LR;                                // Learning rate
    int num_epochs;                         // Number of epochs
    int num_hidden_layers;                  // Number of hidden layers
    std::shared_ptr<InputLayer> inputLayer; // Pointer to the input layer
    std::vector<std::shared_ptr<Layer>> layers; // Layers of the network
    int num_features;                       // Number of features
    double model_loss;                      // Current model loss
    std::vector<double> epoch_losses;       // Losses for each epoch
    std::vector<double> epoch_gradient_norms; // Gradient norms for each epoch
    LossFunction selectedLoss;              // Loss function to be used
    OptimizationMethod optimizationMethod;
    int batch_size; // Batch size

    // Constructor
    NeuralNetwork(float learningrate, int num_epochs, LossFunction lossFunction, int batchSize, OptimizationMethod optimizationMethod);
    NeuralNetwork(const NeuralNetwork& baseNN, const std::vector<double>& encoding);

    // Public methods
    void fit(std::vector<std::vector<double>> featuresMatrix, std::vector<std::vector<double>> labels);
    std::vector<std::vector<double>> getPredictions(std::vector<std::vector<double>> featuresMatrix);
    double calculateLoss(const std::vector<double>& predictions, const std::vector<double>& labels);
    void addLayer(std::shared_ptr<Layer> layer);
    void addInputLayer(std::shared_ptr<InputLayer> inputLayer);
    void reInitializeLayers();
    std::vector<double> getNetworkEncoding() const;
    void setEncoding(std::vector<double> encoding);
};

#endif // NEURALNETWORK_H
