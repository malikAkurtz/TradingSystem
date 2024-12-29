#ifndef CONN_H
#define CONN_H

struct ConnectionGene
{
    int node_in;
    int node_out;
    double weight;
    bool enabled;
    int innovation_number;

    ConnectionGene(int node_in, int node_out, double weight, bool enabled, int innovation_number);
};

struct Connection
{
    int node_in;
    int node_out;
    double weight;
    bool enabled;
    int innovation_number;

    Connection(ConnectionGene connection_gene);
};


#endif