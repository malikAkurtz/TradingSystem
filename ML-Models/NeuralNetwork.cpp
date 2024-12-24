#include "NeuralNetwork.h"
#include "OptimizationTypes.h"

using namespace LinearAlgebra;
using namespace LossFunctions;
using namespace OptimizationMethods;

NeuralNetwork::NeuralNetwork(float learningrate, int num_epochs, LossFunction lossFunction, int batchSize, OptimizationMethod optimizationMethod) : inputLayer(nullptr){
    this->LR = learningrate;
    this->num_hidden_layers = 0;
    this->num_epochs = num_epochs;
    this->selectedLoss = lossFunction;
    this->batch_size = batchSize;
    this->optimizationMethod = optimizationMethod;
}

void NeuralNetwork::fit(std::vector<std::vector<double>> featuresMatrix, std::vector<std::vector<double>>  labels) 
{
    batchGradientDescent(*this, featuresMatrix, labels);
}

// takes in a features matrix and returns a matrix where each column is a vector of 
// predictions for that sample
std::vector<std::vector<double>> NeuralNetwork::getPredictions(std::vector<std::vector<double>> featuresMatrix) 
{
    std::vector<std::vector<double>> featuresMatrix_T = takeTranspose(featuresMatrix);

    printMatrixDebug(featuresMatrix_T);

    // returning a vector of column vectors for each sample that is passed int
    std::vector<std::vector<double>> predictions;

    this->inputLayer->storeInputs(featuresMatrix_T);
    std::vector<std::vector<double>> input_layer_output = this->inputLayer->getInputs();
    printDebug("-------------------Getting Predictions------------------------------");
    printDebug("Input Layer Output");
    printMatrixDebug(input_layer_output);

    std::vector<std::vector<double>> prev_layer_output = input_layer_output;
    for (int i = 0; i < this->num_hidden_layers; i++) {

        this->layers[i]->calculateLayerOutputs(prev_layer_output);

        std::vector<std::vector<double>> this_layer_output = this->layers[i]->getActivationOutputs();

        prev_layer_output = this_layer_output;
        printDebug("This Layer Output");
        printMatrixDebug(this_layer_output);
    }

    predictions = this->layers[num_hidden_layers - 1]->getActivationOutputs();
    return predictions;
}

double NeuralNetwork::calculateLoss(const std::vector<double>& predictions, const std::vector<double>&  labels) 
{
    if (this->selectedLoss == SQUARRED_ERROR) {
        return vectorizedModifiedSquarredError(predictions, labels);
    } else if (this->selectedLoss == BINARY_CROSS_ENTROPY) {
        return vectorizedLogLoss(predictions, labels);
    } else {
        throw std::invalid_argument("NO LOSS FUNCTION SELECTED");
    }
}

void NeuralNetwork::addLayer(std::shared_ptr<Layer> layer) 
{
    this->layers.push_back(layer);
    this->num_hidden_layers += 1;
}

void NeuralNetwork::addInputLayer(std::shared_ptr<InputLayer> inputLayer) {
    this->inputLayer = inputLayer;
}

