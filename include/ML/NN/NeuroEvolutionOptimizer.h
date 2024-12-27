#ifndef NE_OPTIMIZER
#define NE_OPTIMIZER

#include "Optimizer.h"
#include "LossFunctionTypes.h"
#include "LinearAlgebra.h"
#include "NeuroHelperFuncs.h"

class NeuroEvolutionOptimizer : public Optimizer
{
public:
    float mutation_rate;
    int population_size;
    int max_generations;
    LossFunction loss_function;

    NeuroEvolutionOptimizer();

    NeuroEvolutionOptimizer(float mutation_rate, int population_size, int max_generations, LossFunction loss_function);

    void fit(NeuralNetwork &this_network, const std::vector<std::vector<double>>& features_matrix, const std::vector<std::vector<double>>& labels) override;

    double calculateLoss(const std::vector<double> &predictions, const std::vector<double> &labels) override;

};

#endif