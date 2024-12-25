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

// takes in a template base neural net, and an encoding of weights and parameters and constructs a new neural network
NeuralNetwork::NeuralNetwork(const NeuralNetwork& baseNN, const std::vector<double>& encoding)
{
    this->LR = baseNN.LR;
    this->num_hidden_layers = 0;
    this->num_epochs = baseNN.num_epochs;
    this->selectedLoss = baseNN.selectedLoss;
    this->batch_size = baseNN.batch_size;
    this->optimizationMethod = NEUROCHILD;


    int num_neurons_input_layer = baseNN.inputLayer->inputNeurons.size();

    std::shared_ptr<InputLayer> new_input_layer = std::make_shared<InputLayer>(num_neurons_input_layer);
    this->addInputLayer(new_input_layer);

    int encoding_index = 0;
    for (std::shared_ptr<Layer> base_layer : baseNN.layers)
    {   
        // std::cout << "The weights matrix for the base layer is: " << std::endl;
        // std::cout << "[" << std::endl;
        // for (const auto& row : base_layer->getWeightsMatrix()) {
        //     std::cout << "  < ";
        //     for (const auto& elem : row) {
        //         std::cout << elem << " ";
        //     }
        //     std::cout << ">" << std::endl;
        // }
        // std::cout << "]" << std::endl;

        int num_neurons_in_this_layer = base_layer->neurons.size();
        int num_weights_per_neuron = base_layer->neurons[0].getWeights().size() - 1; // since the bias is already included when a layer is initialized

        std::shared_ptr<Layer> new_layer = std::make_shared<Layer>(num_neurons_in_this_layer, num_weights_per_neuron, base_layer->AFtype, base_layer->initalization);

        for (int i = 0; i < num_neurons_in_this_layer; i++)
        {
            for (int j = 0; j < (num_weights_per_neuron); j++)
            {
                // std::cout << "Setting new_layer->neurons[" << i << "].weights[" << j
                //   << "] = encoding[" << encoding_index << "] = "
                //   << encoding[encoding_index] << std::endl;
                new_layer->neurons[i].weights[j] = encoding[encoding_index++];
            }
        }
        // std::cout << "The weights matrix for the copied layer is: " << std::endl;
        // std::cout << "[" << std::endl;
        // for (const auto& row : new_layer->getWeightsMatrix()) {
        //     std::cout << "  < ";
        //     for (const auto& elem : row) {
        //         std::cout << elem << " ";
        //     }
        //     std::cout << ">" << std::endl;
        // }
        // std::cout << "]" << std::endl;

        this->addLayer(new_layer);
    }
}

void NeuralNetwork::fit(std::vector<std::vector<double>> featuresMatrix, std::vector<std::vector<double>>  labels) 
{
    if (this->optimizationMethod == GRADIENT_DESCENT) 
    {
        batchGradientDescent(*this, featuresMatrix, labels);
    }
    else if (this->optimizationMethod == NEUROEVOLUTION)
    {
        NeuroEvolution(*this, featuresMatrix, labels);
    }
    else if (this->optimizationMethod == NEUROCHILD)
    {
        throw std::invalid_argument("This is a child network, can't call fit!");
    }
    else
    {
        throw std::invalid_argument("No Optimization Type Specified!");
    }

}

// takes in a features matrix and returns a matrix where each column is a vector of 
// predictions for that sample
std::vector<std::vector<double>> NeuralNetwork::getPredictions(std::vector<std::vector<double>> featuresMatrix) 
{
    std::vector<std::vector<double>> features_T = takeTranspose(featuresMatrix);

    // returning a vector of column vectors for each sample that is passed int
    std::vector<std::vector<double>> predictions;

    this->inputLayer->storeInputs(features_T);
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
    printDebug("-------------------Got Predictions------------------------------");
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

void NeuralNetwork::reInitializeLayers()
{
    for (std::shared_ptr<Layer> layer : this->layers)
    {
        layer->reInitializeNeurons();
    }
}

std::vector<double> NeuralNetwork::getNetworkEncoding() const
{
    
    std::vector<double> NNsequence;
    for (std::shared_ptr<Layer> layer: this->layers)
    {
        std::vector<double> flattened = flattenMatrix(layer->getWeightsMatrix());
        NNsequence.insert(NNsequence.end(), flattened.begin(), flattened.end());
    }

    return NNsequence;
}

void NeuralNetwork::setEncoding(std::vector<double> encoding)
{
    int num_neurons_prev_layer = this->inputLayer->inputNeurons.size();
    std::shared_ptr<InputLayer> new_input_layer = std::make_shared<InputLayer>(num_neurons_prev_layer);

    int encoding_index = 0;
    for (std::shared_ptr<Layer> base_layer : this->layers)
    {   
        // std::cout << "The weights matrix for the base layer is: " << std::endl;
        // std::cout << "[" << std::endl;
        // for (const auto& row : base_layer->getWeightsMatrix()) {
        //     std::cout << "  < ";
        //     for (const auto& elem : row) {
        //         std::cout << elem << " ";
        //     }
        //     std::cout << ">" << std::endl;
        // }
        // std::cout << "]" << std::endl;


        int num_neurons_cur_layer = base_layer->neurons.size();

        for (int i = 0; i < num_neurons_cur_layer; i++)
        {
            for (int j = 0; j < (num_neurons_prev_layer+1); j++)
            {
                // std::cout << "Setting new_layer->neurons[" << i << "].weights[" << j
                //   << "] = encoding[" << encoding_index << "] = "
                //   << encoding[encoding_index] << std::endl;
                base_layer->neurons[i].weights[j] = encoding[encoding_index++];
            }
        }
        // std::cout << "The weights matrix for the copied layer is: " << std::endl;
        // std::cout << "[" << std::endl;
        // for (const auto& row : new_layer->getWeightsMatrix()) {
        //     std::cout << "  < ";
        //     for (const auto& elem : row) {
        //         std::cout << elem << " ";
        //     }
        //     std::cout << ">" << std::endl;
        // }
        // std::cout << "]" << std::endl;

        num_neurons_prev_layer = num_neurons_cur_layer;
    }
}
