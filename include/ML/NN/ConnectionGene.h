#ifndef CONNECTION_GENE_H
#define CONNECTION_GENE_H

struct ConnectionGene
{
    int node_in;
    int node_out;
    double weight;
    bool enabled;
    int innovation_number;
};

#endif