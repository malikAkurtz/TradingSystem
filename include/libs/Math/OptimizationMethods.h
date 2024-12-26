#ifndef OPT_METHODS
#define OPT_METHODS


#include "../../ML-Models/NN/NetworkLayers.h"
#include "Output.h"
class NeuralNetwork;

namespace OptimizationMethods
{
    void batchGradientDescent(NeuralNetwork &network, const std::vector<std::vector<double>> &featuresMatrix, const std::vector<std::vector<double>> &labels);
    void NeuroEvolution(NeuralNetwork &network);
}

#endif