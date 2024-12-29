#include "Connection.h"


ConnectionGene::ConnectionGene(int node_in, int node_out, double weight, bool enabled, int innovation_number) 
    : node_in(node_in), node_out(node_out), weight(weight), enabled(enabled), innovation_number(innovation_number) {};

Connection::Connection(ConnectionGene connection_gene) 
    : node_in(connection_gene.node_in), node_out(connection_gene.node_out), weight(connection_gene.weight), enabled(connection_gene.enabled), innovation_number(connection_gene.innovation_number) {};

