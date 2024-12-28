#ifndef GENOME_H
#define GENOME_H

#include <vector>
#include "ConnectionGene.h"
#include "NodeGene.h"

struct Genome
{
    std::vector<ConnectionGene> connection_genes;
    std::vector<NodeGene> node_genes;
};

#endif