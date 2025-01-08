#ifndef GENOME_H
#define GENOME_H

#include <vector>
#include "Node.h"
#include <ctime>
#include <iostream>
#include <fstream>
#include <map>
#include "InnovationNum.h"

struct Entity;

struct Genome
{
    std::vector<ConnectionGene> connection_genes;
    std::vector<NodeGene> node_genes;

    Genome();

    Genome(std::vector<ConnectionGene> connection_genes, std::vector<NodeGene> node_genes);

    Genome(int num_input_nodes, int num_output_nodes);

    void mutateAddConnection();

    void mutateAddNode();

    void mutateChangeWeight();

    std::map<int, Node> mapIDtoNode();

    std::map<int, int> mapIDtoDepth();

    void assignConnectionsToNodes(std::map<int, Node> &id_to_node);

    double Genome::calculateCompatibilityDist(const Genome &other_genome, double c1 = 1, double c2 = 1, double c3 = 1) const;

    std::string toString() const;

    std::string toGraphviz() const;

    void saveToDotFile(const std::string &filename) const;

};

#endif