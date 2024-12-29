#ifndef GENOME_H
#define GENOME_H

#include <vector>
#include "Node.h"

struct Genome
{
    std::vector<ConnectionGene> connection_genes;
    std::vector<NodeGene> node_genes;

    Genome(std::vector<ConnectionGene> connection_genes, std::vector<NodeGene> node_genes);
};



#endif