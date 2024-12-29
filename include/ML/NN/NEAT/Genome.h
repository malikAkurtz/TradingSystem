#ifndef GENOME_H
#define GENOME_H

#include <vector>
#include "Node.h"
#include <ctime>
#include <iostream>
#include <fstream>
struct Genome
{
    std::vector<ConnectionGene> connection_genes;
    std::vector<NodeGene> node_genes;

    Genome(std::vector<ConnectionGene> connection_genes, std::vector<NodeGene> node_genes);

    void mutateAddConnection();

    void mutateAddNode();

    std::string toString() const;

    std::string toGraphviz() const;

    void saveToDotFile(const std::string &filename) const;
};

#endif