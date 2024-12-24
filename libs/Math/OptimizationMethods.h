#ifndef OPT_METHODS
#define OPT_METHODS


#include "NeuralNetwork.h"
class NeuralNetwork;

namespace OptimizationMethods
{
    void batchGradientDescent(NeuralNetwork &network, const std::vector<std::vector<double>> &featuresMatrix, const std::vector<std::vector<double>> &labels);
}

#endif