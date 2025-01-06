#include "NeuralNet.h"
#include <iostream>


NeuralNet::NeuralNet()
{
}

NeuralNet::NeuralNet(Genome genome)
{
    this->id_to_node = this->mapIDtoNode(genome);

    this->id_to_depth = this->mapIDtoDepth(genome);

    this->assignConnectionsToNodes(genome);
    this->layers.clear();
    this->assignNodestoLayers();
}

std::map<int, Node> NeuralNet::mapIDtoNode(const Genome& genome)
{
    std::map<int, Node> id_to_node;
    // for every node in the NodeGene sequence of the genome
    for (const NodeGene& node_gene : genome.node_genes)
    {
        // add it to the map which maps ids to nodes
        id_to_node.emplace(node_gene.node_id, Node(node_gene));
        if (node_gene.node_type == HIDDEN)
        {
            id_to_node.at(node_gene.node_id).setActivation(RELU);
        }
        // else if (node_gene.node_type == OUTPUT)
        // {
        //     id_to_node.at(node_gene.node_id).setActivation(TANH);
        // }
    }

    return id_to_node;
}

std::map<int, int> NeuralNet::mapIDtoDepth(const Genome& genome)
{
    std::map<int, int> id_to_depth;

    for (const NodeGene& node_gene : genome.node_genes)
    {
        id_to_depth[node_gene.node_id] = 0;
    }

    for (const NodeGene& node_gene : genome.node_genes)
    {
        if (node_gene.node_type == OUTPUT)
        {
            id_to_depth[node_gene.node_id] = INT_MAX;
        }
    }
    
    bool change_occurred = true;

    while (change_occurred)
    {
        change_occurred = false;
        for (const ConnectionGene &connection_gene : genome.connection_genes)
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
    
    return id_to_depth;
}

void NeuralNet::assignConnectionsToNodes(const Genome& genome)
{
    for (const auto& cg : genome.connection_genes)
    {
        int node_out = cg.node_out;

        auto it = this->id_to_node.find(node_out);
        if (it != this->id_to_node.end()) {
            it->second.connections_in.emplace_back(Connection(cg));
        } else {
            std::cerr << "Error: Node " << node_out << " not found in id_to_node." << std::endl;
        }
    }
}


void NeuralNet::assignNodestoLayers()
{
    int greatest_depth = 0; // the greatest depth not including the output layer


    for (const auto &[key, value] : this->id_to_depth)
    {   
        // if the node isnt an output node, since were initalizing them to infinity, then update the greatest_depth
        if ((this->id_to_node.at(key).node_type != OUTPUT) && (value > greatest_depth))
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

    for (int i = 0; i < greatest_depth + 1; i++) // + 1 to include the output layer
    {
        this->layers.emplace_back(Layer());
    }

    // for every node id mapped to its depth (the map is already sorted )
    for (const auto &[key, value] : this->id_to_depth)
    {   
        if (this->id_to_node.at(key).node_type == OUTPUT)
        {
            this->layers[greatest_depth].node_IDs.push_back(key);
        }
        else
        {
            this->layers[value].node_IDs.push_back(key);
        }
        
    }

}

std::vector<std::vector<double>> NeuralNet::feedForward(const std::vector<std::vector<double>> &features_matrix)
{
    // debugMessage("feedForward", "Beginning Feed Forwad With Feature Matrix: Rows = " + std::to_string(features_matrix.size()) + ", Columns = " + std::to_string(features_matrix[0].size()));
    // debugMessage("feedForward", "Neural Network Before Pass: \n" + this->toString());

    this->loadInputs(features_matrix);

    int last_layer_index = this->layers.size() - 1;

    int num_samples = features_matrix.size();

    std::vector<std::vector<double>> network_outputs(num_samples);

    // starting at 1 to skip the input layer
    for (int l = 1; l < this->layers.size(); l++)
    {
        // debugMessage("feedForward", "Processing Layer: " + std::to_string(l));
        Layer *this_layer = &this->layers[l];
        // for every node in the layer, need to calculate its output and store it in that node
        for (int n = 0; n < this_layer->node_IDs.size(); n++)
        {
            // debugMessage("feedForward", "Processing Node: " + std::to_string(this_layer->node_IDs[n]));
            std::vector<std::vector<double>> scaled_inputs = {};
            // for every connection going into the node
            for (const auto &connection : this->id_to_node.at(this_layer->node_IDs[n]).connections_in)
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
                    // debugMessage("feedForward", "Inputs Coming From Node ID: " + std::to_string(node_in) + " is: ");
                    // printVectorDebug(this->id_to_node.at(node_in).outputs);
                    input = this->id_to_node.at(node_in).outputs;
                }

                std::vector<double> scaled_input = LinearAlgebra::scaleVector(input, connection.weight);
                // printVector(scaled_input);
                scaled_inputs.push_back(scaled_input);
            }
            if (scaled_inputs.empty())
            {
                std::vector<double> default_output(num_samples, 0);
                id_to_node.at(this_layer->node_IDs[n]).storeOutputs(default_output);
                continue;
            }
            std::vector<double> node_output(scaled_inputs[0].size(), 0);

            for (const auto &vector : scaled_inputs)
            {
                // debugMessage("feedForward", "A Vector in Scaled Inputs Looks like: ");
                // printVectorDebug(vector);
                // debugMessage("feedForward", "A Vector in scaled_inputs: ");
                // printVectorDebug(vector);

                node_output = LinearAlgebra::addVectors(node_output, vector);
            }
            // apply acivation
            node_output = id_to_node.at(this_layer->node_IDs[n]).applyActivation(node_output);
            
            id_to_node.at(this_layer->node_IDs[n]).storeOutputs(node_output);
            if (l == last_layer_index) 
            {
                LinearAlgebra::addColumn(network_outputs, node_output);
            }
        }
    }


    // debugMessage("feedForward", "feedForward Result: ");
    // printMatrixDebug(network_outputs);
    return network_outputs;
}

void NeuralNet::loadInputs(const std::vector<std::vector<double>>& features_matrix)
{   

    int num_features = features_matrix[0].size();
    // for every feature column in the features_matrix
    int num_input_nodes = this->layers[0].node_IDs.size() - 1; // not including bias

    // debugMessage("loadInputs", "features_matrix before loading into inputs looks like:");
    // printMatrixDebug(features_matrix);

    if (num_input_nodes != num_features)
    {
        throw std::invalid_argument("The Number of Input Nodes Does Not Match The Number of Features");
    }

    std::vector<Node *> input_nodes = {};

    for (auto& [id, node] : this->id_to_node)
    {
        if (node.node_type == INPUT)
        {
            input_nodes.push_back(&node);
        }
    }

    for (int j = 0; j < num_features; j++) // start
    {
        // save it
        std::vector<double> feature_vector = LinearAlgebra::getColumn(features_matrix, j);
        // debugMessage("loadInputs", "Loading This Vector Into Input Node: ");
        // printVectorDebug(feature_vector);
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
        oss << "<<<<<<<<<Layer " << i << " >>>>>>>>>\n";
        
        for (int j = 0; j < layers[i].node_IDs.size(); j++)
        {
            const Node* node = &this->id_to_node.at(layers[i].node_IDs[j]);
            
            oss << "    Node ID: " << node->node_id << "\n";
            oss << "        Connections In:\n";

            for (int m = 0; m < node->connections_in.size(); m++)
            {
                const Connection& conn = node->connections_in[m];
                oss << "            From: " << conn.node_in
                    << " To: " << conn.node_out << " , " << (conn.enabled ? "true" : "false") <<"\n";
            }
        }
    }
    return oss.str();
}