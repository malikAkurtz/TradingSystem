#include "Genome.h"

Genome::Genome(std::vector<ConnectionGene> connection_genes, std::vector<NodeGene> node_genes) : connection_genes(connection_genes), node_genes(node_genes) {};

double Genome::feedForward(const std::vector<double>& inputs)
{
    
}