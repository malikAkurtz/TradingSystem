#include "NodeType.h"
#include "ActivationFunctionTypes.h"
#include "ActivationFunctions.h"
#include <vector>
#include <map>

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
    std::vector<Layer> layers;

    NeuralNetwork(Genome genome)
    {   
        std::vector<int> input_layer_ids;
        std::vector<int> output_layer_ids;

        std::map<int, Node> id_to_node;
        std::map<int, int> id_to_depth;

        // for every node in the NodeGene sequence of the genome
        for (const NodeGene& node_gene : genome.node_genes)
        {
            // add it to the map which maps ids to nodes
            id_to_node[node_gene.node_id] = Node(node_gene);
            // create a reference to the node
            Node &this_node = id_to_node[node_gene.node_id];

            id_to_depth[this_node.node_id] = 0; // intialize its depth to zero

        }

        calculateLayerDepths(genome, id_to_depth);

        Layer input_layer;
        layers.push_back(input_layer);
        int num_layers = 0;
        for (const auto &[key, value] : id_to_depth)
        {
            if (value == num_layers)
            {
                layers[num_layers].nodes.push_back(id_to_node[key]);
            }
            else
            {
                layers.emplace_back(Layer());
                num_layers++;
                layers[num_layers].nodes.push_back(id_to_node[key]);
            }
            
        }
        }

    void calculateLayerDepths(const Genome& genome, std::map<int, int>& id_to_depth)
    {
        bool change_occurred = true;

        while (change_occurred)
        {
            change_occurred = false;
            for (const ConnectionGene &connection_gene : genome.connection_genes)
            {
                int conn_node_out = connection_gene.node_out;
                int conn_node_in = connection_gene.node_in;

                int prev_conn_node_out_depth = id_to_depth[conn_node_out];

                id_to_depth[conn_node_out] = std::max(id_to_depth[conn_node_out], id_to_depth[conn_node_in] + 1);

                if (prev_conn_node_out_depth < id_to_depth[conn_node_out])
                {
                    change_occurred = true;
                }
            }
        }

    }
};