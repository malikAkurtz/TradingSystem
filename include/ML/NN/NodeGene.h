#ifndef NODE_GENE_H
#define NODE_GENE_H

#include "NodeType.h"

struct NodeGene
{
    int node_id;
    NodeType node_type;

    NodeGene(int node_id, NodeType node_type);
};

#endif