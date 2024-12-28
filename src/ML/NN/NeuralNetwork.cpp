#include "NeuralNetwork.h"
#include "Optimizer.h"

using namespace LinearAlgebra;
using namespace LossFunctions;

NeuralNetwork::NeuralNetwork() : optimizer(nullptr)
{
    this->num_hidden_layers = 0;
    this->model_loss = INFINITY;
}

NeuralNetwork::NeuralNetwork(Optimizer* optimizer) : optimizer(optimizer)
{
    this->num_hidden_layers = 0;
    this->model_loss = INFINITY;
}


// takes in a template base neural net, and an encoding of weights and parameters and constructs a new neural network
NeuralNetwork::NeuralNetwork(const NeuralNetwork& base_NN, const std::vector<double>& encoding)
{
    this->num_hidden_layers = 0;
    this->layers.clear();

    int num_nodes_input_layer = base_NN.input_layer.input_nodes.size();

    this->input_layer = InputLayer(num_nodes_input_layer);

    int num_nodes_in_previous_layer = num_nodes_input_layer;
    int encoding_index = 0;
    for (Layer base_layer : base_NN.layers)
    {   
        int num_nodes_in_this_layer = base_layer.nodes.size();

        Layer new_layer = Layer(num_nodes_in_this_layer, num_nodes_in_previous_layer, base_layer.activation_function, base_layer.node_initalization);

        for (int i = 0; i < num_nodes_in_this_layer; i++)
        {
            for (int j = 0; j < (num_nodes_in_previous_layer+1); j++) // need to include the bias here
            {

                new_layer.nodes[i].weights[j] = encoding[encoding_index++];
            }
        }

        this->layers.push_back(new_layer);
        this->num_hidden_layers += 1;
        num_nodes_in_previous_layer = new_layer.nodes.size();
    }
}

double NeuralNetwork::calculateFinalModelLoss(std::vector<std::vector<double>> features_matrix, std::vector<std::vector<double>> labels) 
{
    // now getting predictions of the entire feature matrix, i.e all samples
    // best_predictions will then consist of a vector of column vectors
    std::vector<std::vector<double>> best_predictions = this->feedForward(features_matrix);
    // printDebug("Number of predictions");
    // printDebug(best_predictions.size());
    // printDebug("looks like");
    // printMatrixDebug(best_predictions);
    std::vector<std::vector<double>> labels_T = takeTranspose(labels);
    // printDebug("Number of labels");
    // printDebug(labels_T.size());
    // printDebug("looks like");
    // printMatrixDebug(labels_T);
    double accumulated_final_model_loss = 0;
    // printDebug("best_predictions");
    // printMatrixDebug(best_predictions);
    // printDebug("labels_T");
    // printMatrixDebug(labels_T);
    for (int i = 0; i < best_predictions[0].size(); i++)
    {
        accumulated_final_model_loss += this->optimizer->calculateLoss(getColumn(best_predictions, i), getColumn(labels_T, i));
        // printDebug("Loss for these samples");
        // printDebug(this->calculateLoss(getColumn(best_predictions, i), getColumn(labels_T, i)));
    }
    this->model_loss = accumulated_final_model_loss / labels_T[0].size();
    return this->model_loss;
}

void NeuralNetwork::fit(const std::vector<std::vector<double>>& features_matrix, const std::vector<std::vector<double>>& labels)
{
    if (!optimizer) 
    {
        throw std::invalid_argument("No Optimizer Selected!");
    }
    optimizer->fit(*this, features_matrix, labels);
    calculateFinalModelLoss(features_matrix, labels);
}

// takes in a features matrix and returns a matrix where each column is a vector of 
// predictions for that sample
std::vector<std::vector<double>> NeuralNetwork::feedForward(std::vector<std::vector<double>> features_matrix) 
{
    std::vector<std::vector<double>> features_T = takeTranspose(features_matrix);

    // returning a vector of column vectors for each sample that is passed int
    std::vector<std::vector<double>> predictions;

    this->input_layer.storeInputs(features_T);
    std::vector<std::vector<double>> input_layer_output = this->input_layer.getInputs();
    printDebug("-------------------Getting Predictions------------------------------");
    printDebug("Input Layer Output");
    printMatrixDebug(input_layer_output);

    std::vector<std::vector<double>> prev_layer_output = input_layer_output;
    for (int i = 0; i < this->num_hidden_layers; i++) {
        printDebug("Here");
        this->layers[i].calculateLayerOutputs(prev_layer_output);

        std::vector<std::vector<double>> this_layer_output = this->layers[i].getActivationOutputs();

        prev_layer_output = this_layer_output;
        printDebug("This Layer Output");
        printMatrixDebug(this_layer_output);
    }

    predictions = this->layers[num_hidden_layers - 1].getActivationOutputs();
    printDebug("-------------------Got Predictions------------------------------");
    return predictions;
}

void NeuralNetwork::addInputLayer(int num_features) {
    this->input_layer = InputLayer(num_features);
    this->layer_sizes.push_back(num_features);
}

void NeuralNetwork::addLayer(int num_nodes, ActivationFunctionType activation_function, NodeInitializationType neuron_initialization) 
{
    this->layers.emplace_back(Layer(num_nodes, this->layer_sizes.back(), activation_function, neuron_initialization));
    this->layer_sizes.push_back(num_nodes);
    this->num_hidden_layers += 1;
}


void NeuralNetwork::reInitializeLayers()
{
    for (Layer layer : this->layers)
    {
        layer.reInitializeNodes();
    }
}

std::vector<double> NeuralNetwork::getNetworkEncoding() const
{
    
    std::vector<double> encoding;
    for (Layer layer: this->layers)
    {
        std::vector<double> flattened = flattenMatrix(layer.getWeightsMatrix());
        encoding.insert(encoding.end(), flattened.begin(), flattened.end());
    }

    return encoding;
}

void NeuralNetwork::setEncoding(std::vector<double> encoding)
{
    int num_nodes_prev_layer = this->input_layer.input_nodes.size();
    this->input_layer = InputLayer(num_nodes_prev_layer);

    int encoding_index = 0;
    for (int n = 0; n < this->layers.size(); n++)
    {   
        int num_nodes_cur_layer = this->layers[n].nodes.size();

        for (int i = 0; i < num_nodes_cur_layer; i++)
        {
            for (int j = 0; j < (num_nodes_prev_layer+1); j++)
            {
                this->layers[n].nodes[i].weights[j] = encoding[encoding_index++];
            }
        }

        num_nodes_prev_layer = num_nodes_cur_layer;
    }
}
