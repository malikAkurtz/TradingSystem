#include "NeuralNet.h"
#include <iostream>


NeuralNet::NeuralNet()
{
}

NeuralNet::NeuralNet(Genome genome)
{
    
    this->id_to_node = genome.mapIDtoNode();
    this->id_to_depth = genome.mapIDtoDepth();

    genome.assignConnectionsToNodes(this->id_to_node);
    this->assignNodestoLayers();
}

NeuralNet::~NeuralNet()
{
    for (auto& [id, node] : this->id_to_node)
    {
        delete node;
    }
    this->id_to_node.clear();
}


void NeuralNet::assignNodestoLayers()
{
    int greatest_depth = 0;
    for (const auto &[key, value] : this->id_to_depth)
    {
        if (value > greatest_depth)
        {
            greatest_depth = value;
        }
    }
    greatest_depth++; // bc of index 0

    for (int i = 0; i < greatest_depth; i++)
    {
        this->layers.emplace_back(Layer());
    }

    // for every node id mapped to its depth (the map is already sorted )
    for (const auto &[key, value] : this->id_to_depth)
    {
        this->layers[value].nodes.push_back(this->id_to_node.at(key));
    }
}


std::vector<std::vector<double>> NeuralNet::feedForward(const std::vector<std::vector<double>> &features_matrix)
{
    this->loadInputs(features_matrix);

    int last_layer_index = this->layers.size() - 1;

    std::vector<std::vector<double>> network_outputs;

    // starting at 1 to skip the input layer
    std::cout << "----------------------PROCESSING LAYERS----------------------" << std::endl;
    for (int l = 1; l < this->layers.size(); l++)
    {
        std::cout << "Processing Layer: " << l << std::endl;
        Layer &this_layer = this->layers[l];
        // for every node in the layer, need to calculate its output and store it in that node
        for (int n = 0; n < this_layer.nodes.size(); n++)
        {
            std::cout << "Processing Node: " << this_layer.nodes[n]->node_id << std::endl;
            std::vector<std::vector<double>> scaled_inputs;
            // for every connection going into the node
            for (const auto &connection : this_layer.nodes[n]->connections_in)
            {    
                if (!connection.enabled) 
                {
                    continue;
                }
                int node_in = connection.node_in;
                std::cout << "Node In ID: " << node_in << std::endl;
                // get the incoming nodes outputs
                std::vector<double> input = this->id_to_node.at(node_in)->outputs;
                std::cout << "Has Output Vector: " << std::endl;
                printVector(input);
                std::vector<double> scaled_input = LinearAlgebra::scaleVector(input, connection.weight);
                std::cout << "Has Scaled Outputs: " << std::endl;
                printVector(scaled_input);
                scaled_inputs.push_back(scaled_input);
            }
            std::vector<double> node_output(scaled_inputs[0].size(), 0);
            for (const auto& vector: scaled_inputs)
            {
                node_output = LinearAlgebra::addVectors(node_output, vector);
            }
            // apply acivation
            node_output = this_layer.nodes[n]->applyActivation(node_output);
            std::cout << "After Summing All Vectors, Output is: " << std::endl;
            printVector(node_output);
            this_layer.nodes[n]->storeOutputs(node_output);
            if (l == last_layer_index) 
            {
                network_outputs.push_back(node_output);
            }
        }
    }
    return network_outputs;
}

void NeuralNet::loadInputs(const std::vector<std::vector<double>>& features_matrix)
{
    int num_features = features_matrix[0].size();
    // for every feature column in the features_matrix
    std::cout << "----------------------LOADING INPUTS----------------------" << std::endl;
    for (int j = 0; j < num_features; j++)
    {
        std::cout << "Loading Feature: " << j << " Into Input Layer" << std::endl;
        // save it
        std::vector<double> feature_vector = LinearAlgebra::getColumn(features_matrix, j);
        // store it in the input neurons in the input layer (layers index 0)
        this->layers[0].nodes[j]->storeOutputs(feature_vector);
    }
    for (const auto& node : this->layers[0].nodes)
    {
        std::cout << "Input Node: " << node->node_id << " Has Output Vector" << std::endl;
        printVector(node->outputs);
    }
    std::cout << "----------------------DONE LOADING INPUTS----------------------" << std::endl;
}

