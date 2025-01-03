#include "NeuralNet.h"
#include <iostream>


NeuralNet::NeuralNet()
{
}

NeuralNet::NeuralNet(Genome genome)
{
    this->id_to_node.clear();
    this->id_to_node = genome.mapIDtoNode();

    // std::ostringstream oss;
    // oss << "id_to_node Map:\n";
    // for (const auto& pair : id_to_node)
    // {
    //     oss << "  Node ID: " << pair.first << ", Node ID: " << pair.second.node_id << "\n";
    // }
    // debugMessage("NeuralNet Constructer", "\n" + oss.str());

    this->id_to_depth.clear();
    this->id_to_depth = genome.mapIDtoDepth();

    genome.assignConnectionsToNodes(this->id_to_node);
    this->layers.clear();
    this->assignNodestoLayers();
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

    std::ostringstream oss;
    // oss << "id_to_depth Map:\n";
    // for (const auto& pair : id_to_depth)
    // {
    //     oss << "  Node ID: " << pair.first << ", Depth: " << pair.second << "\n";
    // }

    // debugMessage("assignNodestoLayers", "Node/Layer Depths Before Assignment:\n " + oss.str());

    for (int i = 0; i < greatest_depth; i++)
    {
        this->layers.emplace_back(Layer());
    }

    // for every node id mapped to its depth (the map is already sorted )
    for (const auto &[key, value] : this->id_to_depth)
    {
        this->layers[value].nodes.push_back(&this->id_to_node.at(key));
    }

}

std::vector<std::vector<double>> NeuralNet::feedForward(const std::vector<std::vector<double>> &features_matrix)
{
    debugMessage("feedForward", "Beginning Feed Forwad With Feature Matrix: Rows = " + std::to_string(features_matrix.size()) + ", Columns = " + std::to_string(features_matrix[0].size()));

    this->loadInputs(features_matrix);

    //std::cout << "Made it after load inputs" << std::endl;
    int last_layer_index = this->layers.size() - 1;

    int num_samples = features_matrix.size();
    // int num_labels = this->layers[last_layer_index].nodes.size();
    // std::cout << "num_labels: " << num_labels << std::endl;

    std::vector<std::vector<double>> network_outputs(num_samples);

    // starting at 1 to skip the input layer
    for (int l = 1; l < this->layers.size(); l++)
    {
        Layer &this_layer = this->layers[l];
        // for every node in the layer, need to calculate its output and store it in that node
        for (int n = 0; n < this_layer.nodes.size(); n++)
        {
            std::vector<std::vector<double>> scaled_inputs;
            // for every connection going into the node
            for (const auto &connection : this_layer.nodes[n]->connections_in)
            {    
                // if the connection is disabled, skip it
                if (!connection.enabled) 
                {
                    continue;
                }

                int node_in = connection.node_in;
                std::vector<double> input;
                // get the incoming nodes outputs
                if (connection.node_in == -1) // bias
                {
                    input = std::vector<double>(num_samples, 1);
                }
                else
                {
                    input = this->id_to_node.at(node_in).outputs;
                }

                std::vector<double> scaled_input = LinearAlgebra::scaleVector(input, connection.weight);
                // printVector(scaled_input);
                scaled_inputs.push_back(scaled_input);
            }
            if (scaled_inputs.empty())
            {
                std::vector<double> default_output(num_samples, 0);
                this_layer.nodes[n]->storeOutputs(default_output);
                continue;
            }
            std::vector<double> node_output(scaled_inputs[0].size(), 0);

            for (const auto &vector : scaled_inputs)
            {
                // debugMessage("feedForward", "A Vector in Scaled Inputs Looks like: ");
                // printVectorDebug(vector);
                node_output = LinearAlgebra::addVectors(node_output, vector);
            }
            // apply acivation
            node_output = this_layer.nodes[n]->applyActivation(node_output);
            
            this_layer.nodes[n]->storeOutputs(node_output);
            if (l == last_layer_index) 
            {
                LinearAlgebra::addColumn(network_outputs, node_output);
            }
        }
    }

    // std::cout << "Networks Outputs are:" << std::endl;
    // printMatrix(network_outputs);
    debugMessage("feedForward", "feedForward Result: ");
    printMatrixDebug(network_outputs);
    return network_outputs;
}

void NeuralNet::loadInputs(const std::vector<std::vector<double>>& features_matrix)
{   

    int num_features = features_matrix[0].size();
    // for every feature column in the features_matrix
    int num_nodes_w_bias = this->layers[0].nodes.size();
    int num_input_nodes = this->layers[0].nodes.size() - 1; // not including bias

    if (num_input_nodes != num_features)
    {
        throw std::invalid_argument("The Number of Input Nodes Does Not Match The Number of Features");
    }

    std::vector<Node *> input_nodes;

    for (Node* node : this->layers[0].nodes)
    {
        if (node->node_type == INPUT)
        {
            input_nodes.push_back(node);
        }
    }

    for (int j = 0; j < num_features; j++) // start
    {
        // save it
        std::vector<double> feature_vector = LinearAlgebra::getColumn(features_matrix, j);

        input_nodes[j]->storeOutputs(feature_vector);
    }
}


////////////////////////////////////////////////////////////////////////////////////////////////////

  // for std::ostringstream

std::string NeuralNet::toString() const
{
    std::ostringstream oss;

    for (int i = 0; i < layers.size(); i++)
    {
        oss << "Layer: " << i << " Consists of:\n";
        
        for (int j = 0; j < layers[i].nodes.size(); j++)
        {
            Node* node = layers[i].nodes[j];
            
            oss << "  Node ID: " << node->node_id << "\n";
            oss << "  Has Connections:\n";

            for (int m = 0; m < node->connections_in.size(); m++)
            {
                const Connection& conn = node->connections_in[m];
                oss << "    From: " << conn.node_in 
                    << " To: "   << conn.node_out << "\n";
            }
        }
    }
    return oss.str();
}