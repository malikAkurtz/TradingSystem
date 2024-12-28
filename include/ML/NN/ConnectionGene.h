#ifndef CONNECTION_GENE_H
#define CONNECTION_GENE_H

struct ConnectionGene
{
    int node_out;
    int node_in;
    double weight;
    bool enabled;
    int innovation_number;

    ConnectionGene(int node_in, int node_out, double weight, bool enabled, int innovation_number);
};

#endif