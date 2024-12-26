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

    virtual void fit(NeuralNetwork &thisNetwork, const std::vector<std::vector<double>>& featuresMatrix, const std::vector<std::vector<double>>& labels) = 0;

    virtual double calculateLoss(const std::vector<double> &predictions, const std::vector<double> &labels) = 0;

    virtual ~Optimizer() {};
};

////////////////////////////////////////////////////////////////////////////////////////////////////
class NeuroEvolutionOptimizer : public Optimizer
{
public:
    float mutationRate;
    int populationSize;
    int maxGenerations;
    LossFunction lossFunction;

    NeuroEvolutionOptimizer();

    NeuroEvolutionOptimizer(float mutationRate, int populationSize, int maxGenerations, LossFunction lossFunction);

    void fit(NeuralNetwork &thisNetwork, const std::vector<std::vector<double>>& featuresMatrix, const std::vector<std::vector<double>>& labels) override;

    double calculateLoss(const std::vector<double> &predictions, const std::vector<double> &labels) override;
};

class GradientDescentOptimizer : public Optimizer
{
public:
    float learningRate;
    int numEpochs;
    int batchSize;
    LossFunction lossFunction;

    std::vector<double> epochLosses;
    std::vector<double> gradientNorms;

    GradientDescentOptimizer();

    GradientDescentOptimizer(float learningRate, int numEpochs, int batchSize, LossFunction lossFunction);

    void fit(NeuralNetwork &thisNetwork, const std::vector<std::vector<double>>& featuresMatrix, const std::vector<std::vector<double>>& labels) override;

    double calculateLoss(const std::vector<double> &predictions, const std::vector<double> &labels) override;

};


#endif