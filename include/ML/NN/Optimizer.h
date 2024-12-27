#ifndef OPTIMIZER
#define OPTIMIZER

#include <vector>

class NeuralNetwork;

class Optimizer
{
public:

    virtual void fit(NeuralNetwork &this_network, const std::vector<std::vector<double>>& features_matrix, const std::vector<std::vector<double>>& labels) = 0;

    virtual double calculateLoss(const std::vector<double> &predictions, const std::vector<double> &labels) = 0;

    virtual ~Optimizer() {};
};

////////////////////////////////////////////////////////////////////////////////////////////////////


#endif