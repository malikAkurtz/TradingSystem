#include "NeuralNet.h"
#include <iostream>

NeuralNet::NeuralNet(Genome genome)
{   
    std::map<int, int> id_to_depth;

    // for every node in the NodeGene sequence of the genome
    for (const NodeGene& node_gene : genome.node_genes)
    {
        // add it to the map which maps ids to nodes
        this->id_to_node[node_gene.node_id] = new Node(node_gene);

        id_to_depth[node_gene.node_id] = 0; // intialize its depth to zero

    }

    this->calculateLayerDepths(genome.connection_genes, id_to_depth);
    this->assignConnectionstoNodes(genome.connection_genes);
    this->assignNodestoLayers(id_to_depth);

}


void NeuralNet::calculateLayerDepths(const std::vector<ConnectionGene>& connection_genes, std::map<int, int>& id_to_depth)
{
    bool change_occurred = true;

    while (change_occurred)
    {
        change_occurred = false;
        for (const ConnectionGene &connection_gene : connection_genes)
        {   
            int conn_node_in = connection_gene.node_in; 
            int conn_node_out = connection_gene.node_out;
            
            int prev_conn_node_out_depth = id_to_depth[conn_node_out];

            id_to_depth[conn_node_out] = std::max(id_to_depth[conn_node_out], id_to_depth[conn_node_in] + 1);

            if (prev_conn_node_out_depth < id_to_depth[conn_node_out])
            {
                change_occurred = true;
            }
        }
    }
}

void NeuralNet::assignNodestoLayers(const std::map<int, int>& id_to_depth)
{
    int greatest_depth = 0;
    for (const auto &[key, value] : id_to_depth)
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
    for (const auto &[key, value] : id_to_depth)
    {
        this->layers[value].nodes.push_back(this->id_to_node.at(key));
    }
}

void NeuralNet::assignConnectionstoNodes(const std::vector<ConnectionGene>& connection_genes)
{
    for (const auto& cg : connection_genes)
    {
        int node_out = cg.node_out;
        this->id_to_node[node_out]->connections_in.emplace_back(Connection(cg));
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