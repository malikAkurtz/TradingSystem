#include "/Users/malikkurtz/Coding/TradingSystem/include/ML/NN/NEAT/Node.h"


NodeGene::NodeGene(int node_id, NodeType node_type) 
    : node_id(node_id), node_type(node_type) {};


std::string nodeTypeToString(NodeType type)
{
    switch (type)
    {
        case INPUT: return "INPUT";
        case HIDDEN: return "HIDDEN";
        case OUTPUT: return "OUTPUT";
        case BIAS: return "BIAS";
        default:
            return "UNKNOWN";
    }
}

std::string NodeGene::toString() const
{
    return "NodeGene(node_id: " + std::to_string(this->node_id) +
           ", node_type: " + nodeTypeToString(this->node_type) + ")";
}



//Node::Node() : node_id(-1), node_type(INPUT), activation(RELU) {};

Node::Node(NodeGene node_gene) 
    : node_id(node_gene.node_id), node_type(node_gene.node_type) {};

double Node::calculateNodeOutput(const std::vector<double>& input_vector)
{
    double cum_sum = 0;
    for (int i = 0; i < input_vector.size(); i++)
    {
        if (connections_in[i].enabled)
        {
            cum_sum += input_vector[i] * connections_in[i].weight;
        }
    }

    return ActivationFunctions::ReLU(cum_sum);
}

void Node::storeOutputs(std::vector<double> outputs)
{
    this->outputs = outputs;
}

std::vector<double> Node::applyActivation(const std::vector<double>& pre_activation_outputs)
{
    std::vector<double> acivation_output;
    if (this->activation == RELU)
    {
        return ActivationFunctions::vectorReLU(pre_activation_outputs);
    }
    else if (this->activation == SIGMOID)
    {
        return ActivationFunctions::vectorSigmoid(pre_activation_outputs);
    }
    else
    {
        return pre_activation_outputs;
    }
}