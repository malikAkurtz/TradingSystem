#ifndef NEURALNET_H
#define NEURALNET_H

#include <vector>
#include <map>
#include "Node.h"
#include "Genome.h"
#include "/Users/malikkurtz/Coding/TradingSystem/include/libs/Math/LinearAlgebra.h"
#include "Output.h"

struct Layer
{
public:
    std::vector<Node*> nodes;
};

class NeuralNet
{
public:
    std::vector<Layer> layers;
    std::map<int, Node*> id_to_node;
    std::map<int, int> id_to_depth;

    NeuralNet(Genome genome);

    void assignNodestoLayers(const std::map<int, int> &id_to_depth);

    // want to return a matrix where each row is the networks predictions for that sample
    // the matrix allows us to support batches
    std::vector<std::vector<double>> feedForward(const std::vector<std::vector<double>> &features_matrix);

    void loadInputs(const std::vector<std::vector<double>>& features_matrix);
};

#endif