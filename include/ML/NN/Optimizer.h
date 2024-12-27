#ifndef OPTIMIZER
#define OPTIMIZER

#include "LossFunctions.h"
#include "LinearAlgebra.h"
#include "NeuroHelperFuncs.h"
#include "LossFunctions.h"

class NeuralNetwork;

class Optimizer
{
public:

    virtual void fit(NeuralNetwork &this_network, const std::vector<std::vector<double>>& features_matrix, const std::vector<std::vector<double>>& labels) = 0;

    virtual double calculateLoss(const std::vector<double> &predictions, const std::vector<double> &labels) = 0;

    virtual ~Optimizer() {};
};

////////////////////////////////////////////////////////////////////////////////////////////////////
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

class GradientDescentOptimizer : public Optimizer
{
public:
    float learning_rate;
    int num_epochs;
    int batch_size;
    LossFunction loss_function;
    std::vector<double> epoch_losses;
    std::vector<double> gradient_norms;

    GradientDescentOptimizer();

    GradientDescentOptimizer(float learning_rate, int num_epochs, int batch_size, LossFunction loss_function);

    void fit(NeuralNetwork &thisNetwork, const std::vector<std::vector<double>>& features_matrix, const std::vector<std::vector<double>>& labels) override;

    double calculateLoss(const std::vector<double> &predictions, const std::vector<double> &labels) override;

};


#endif