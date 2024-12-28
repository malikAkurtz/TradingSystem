#include "NodeType.h"
#include "ActivationFunctionTypes.h"
#include "ActivationFunctions.h"
#include <vector>

struct Connection
{
    int node_in;
    int node_out;
    double weight;
    bool enabled;
    int innovation_number;

    Connection(ConnectionGene connection_gene) : node_in(connection_gene.node_in), node_out(connection_gene.node_out), weight(connection_gene.weight), enabled(connection_gene.enabled), innovation_number(connection_gene.innovation_number) {};
};

struct ConnectionGene
{
    int node_in;
    int node_out;
    double weight;
    bool enabled;
    int innovation_number;

    ConnectionGene(int node_in, int node_out, double weight, bool enabled, int innovation_number) : node_in(node_in), node_out(node_out), weight(weight), enabled(enabled), innovation_number(innovation_number) {};
};

class Node
{
public:
    int node_id;
    NodeType node_type;
    ActivationFunctionType activation = RELU;
    std::vector<Connection> connections_in;

    Node(NodeGene node_gene) : node_id(node_gene.node_id), node_type(node_gene.node_type) {};

    double calculateNodeOutput(std::vector<double> input_vector)
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
};

struct NodeGene
{
    int node_id;
    NodeType node_type;

    NodeGene(int node_id, NodeType node_type) : node_id(node_id), node_type(node_type) {};
};

struct Genome
{
    std::vector<ConnectionGene> connection_genes;
    std::vector<NodeGene> node_genes;

    Genome(std::vector<ConnectionGene> connection_genes, std::vector<NodeGene> node_genes) : connection_genes(connection_genes), node_genes(node_genes) {};

};

struct Layer
{
public:
    std::vector<Node> nodes;

};

class NeuralNetwork
{
public:
    std::vector<Layer> layer;

    NeuralNetwork(Genome genome)
    {   
        // start by organizing nodes into layers
        // an input layer is essentialy just a collection of nodes which have no incoming connections, defined by having an id of -1
        Layer input_layer;
        std::vector<Layer> hidden_layers;
        Layer output_layer;
        for (NodeGene node_gene : genome.node_genes)
        {
            if (node_gene.node_type == INPUT)
            {
                input_layer.nodes.emplace_back(Node(node_gene));
            }
            else if (node_gene.node_type == OUTPUT)
            {
                output_layer.nodes.emplace_back(Node(node_gene));
            }
        }
    }
};