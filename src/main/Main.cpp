#include "NeuralNet.h"
#include "Output.h"
#include <chrono>
#include <thread>
#include <cstdlib>

bool DEBUG = false;

int global_innovation_number = 0;

int main()
{
    srand(time(0)); 

    NodeGene ng1(1, INPUT);
    NodeGene ng2(2, INPUT);
    NodeGene ng3(3, INPUT);
    NodeGene ng4(4, OUTPUT);
    NodeGene ng5(5, HIDDEN);
    std::vector<NodeGene> node_genes = {ng1, ng2, ng3, ng4, ng5};

    ConnectionGene cg1(1, 4, 0.7, true, 1);
    global_innovation_number++;
    ConnectionGene cg2(2, 4, 0.5, false, 2);
    global_innovation_number++;
    ConnectionGene cg3(3, 4, 0.5, true, 3);
    global_innovation_number++;
    ConnectionGene cg4(2, 5, 0.2, true, 4);
    global_innovation_number++;
    ConnectionGene cg5(5, 4, 0.4, true, 5);
    global_innovation_number++;
    ConnectionGene cg6(1, 5, 0.6, true, 6);
    global_innovation_number++;


    std::vector<ConnectionGene> connection_genes = {cg1, cg2, cg3, cg4, cg5, cg6};

    Genome genome(connection_genes, node_genes);
    
    // while (true)
    // {
    
    

    // genome.mutateAddConnection();
    // genome.mutateAddConnection();
    // genome.mutateAddConnection();
    // genome.mutateAddConnection();
    // genome.mutateAddNode();

    // genome.saveToDotFile("genome_graph.dot");

    // print(genome.toString());

    // std::system("python3 graph_node.py");

    // std::this_thread::sleep_for(std::chrono::milliseconds(100));
    // }
    print(genome.toString());

    genome.mutateAddNode();
    genome.mutateAddNode();
    genome.mutateAddNode();
    genome.mutateAddNode();
    genome.mutateAddNode();
    genome.mutateAddNode();
    genome.mutateAddNode();
    genome.mutateAddNode();
    genome.mutateAddNode();
    genome.mutateAddNode();
    genome.mutateAddNode();
    genome.mutateAddNode();
    genome.mutateAddNode();
    genome.mutateAddNode();
    genome.mutateAddNode();
    genome.mutateAddNode();
    genome.mutateAddNode();
    genome.mutateAddNode();
    genome.mutateAddNode();
    genome.mutateAddNode();
    genome.mutateAddNode();
    genome.mutateAddNode();
    genome.mutateAddNode();
    genome.mutateAddNode();
    genome.mutateAddNode();
    genome.mutateAddNode();
    genome.mutateAddNode();
    genome.mutateAddNode();
    genome.mutateAddNode();
    genome.mutateAddNode();
    genome.mutateAddNode();
    genome.mutateAddNode();
    genome.mutateAddNode();
    genome.mutateAddNode();
    genome.mutateAddNode();
    genome.mutateAddNode();
    

    // genome.mutateAddConnection();
    // genome.mutateAddConnection();
    // genome.mutateAddConnection();
    // genome.mutateAddConnection();
    // genome.mutateAddConnection();
    // genome.mutateAddConnection();


    print(genome.toString());

    std::cout << "Made it Here" << std::endl;
    genome.saveToDotFile("genome_graph.dot");

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


    std::cout << "About To Feed Forward" << std::endl;
    std::vector<std::vector<double>> network_outputs = network.feedForward({{1, 2, 3}, {4, 5, 6}});
    printMatrix(network_outputs);

    return 0;
}