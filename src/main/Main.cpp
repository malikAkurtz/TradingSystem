#include "NeuralNet.h"
#include "Output.h"
#include <chrono>
#include <thread>
#include <cstdlib>
#include "Entity.h"

bool DEBUG = false;

int global_innovation_number = 0;

int main1()
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

    while (true) 
    {   
        if (rand() % 2)
        {
            genome.mutateAddConnection();
        }
        else
        {
            genome.mutateAddNode();
        }
        std::this_thread::sleep_for(std::chrono::milliseconds(50));
        genome.saveToDotFile("genome_graph.dot");
        std::system("python3 graph_node.py");
    }

    print(genome.toString());

    

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

int main()
{
    srand(time(0)); 

    std::vector<ConnectionGene> parent1_connection_genes = {
        ConnectionGene(1, 4, 1, true, 1),
        ConnectionGene(2, 4, 1, true, 2),
        ConnectionGene(3, 4, 1, true, 3),
        ConnectionGene(2, 5, 1, true, 4),
        ConnectionGene(5, 4, 1, true, 5),
        ConnectionGene(1, 5, 1, true, 8),
    };

    std::vector<ConnectionGene> parent2_connection_genes = {
        ConnectionGene(1, 4, 0, true, 1),
        ConnectionGene(2, 4, 0, true, 2),
        ConnectionGene(3, 4, 0, true, 3),
        ConnectionGene(2, 5, 0, true, 4),
        ConnectionGene(5, 4, 0, true, 5),
        ConnectionGene(5, 6, 0, true, 6),
        ConnectionGene(6, 4, 0, true, 7),
        ConnectionGene(3, 5, 0, true, 9),
        ConnectionGene(1, 6, 0, true, 10),
    };

    Entity parent1;
    parent1.genome.connection_genes = parent1_connection_genes;
    parent1.fitness = 8;

    Entity parent2;
    parent2.genome.connection_genes = parent2_connection_genes;
    parent2.fitness = 9;

    Genome offspring = parent1.crossover(parent2);

    print(offspring.toString());

    return 0;
}