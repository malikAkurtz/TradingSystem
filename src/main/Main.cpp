#include "NeuralNet.h"
#include "Output.h"

bool DEBUG = false;

int main()
{

    NodeGene ng1(1, INPUT);
    NodeGene ng2(2, INPUT);
    NodeGene ng3(3, INPUT);
    NodeGene ng4(4, OUTPUT);
    NodeGene ng5(5, HIDDEN);
    std::vector<NodeGene> node_genes = {ng1, ng2, ng3, ng4, ng5};

    ConnectionGene cg1(1, 4, 0.7, true, 1);
    ConnectionGene cg2(2, 4, 0.5, false, 2);
    ConnectionGene cg3(3, 4, 0.5, true, 3);
    ConnectionGene cg4(2, 5, 0.2, true, 4);
    ConnectionGene cg5(5, 4, 0.4, true, 5);
    ConnectionGene cg6(1, 5, 0.6, true, 6);

    std::vector<ConnectionGene> connection_genes = {cg1, cg2, cg3, cg4, cg5, cg6};

    Genome genome(connection_genes, node_genes);

    NeuralNet network(genome);

    for (int i = 0; i < network.layers.size(); i++)
    {
        std::cout << "Layer: " << i << " Consists of:" << std::endl;
        for (int j = 0; j < network.layers[i].nodes.size(); j++)
        {
            std::cout << network.layers[i].nodes[j]->node_id << std::endl;
            std::cout << "Has Connections: " << std::endl;
            for (int m = 0; m < network.layers[i].nodes[j]->connections_in.size(); m++)
            {
                std::cout << "From: " << network.layers[i].nodes[j]->connections_in[m].node_in << " To: " << network.layers[i].nodes[j]->connections_in[m].node_out << std::endl;
            }
        }
    }

    std::vector<std::vector<double>> network_outputs = network.feedForward({{1, 2, 3}, {4, 5, 6}});
    printMatrix(network_outputs);

    return 0;
}