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
private:
    float mutationRate;
    int populationSize;
    int maxGenerations;
    LossFunction lossFunction;

public:
    NeuroEvolutionOptimizer();

    NeuroEvolutionOptimizer(float mutationRate, int populationSize, int maxGenerations, LossFunction lossFunction);

    void fit(NeuralNetwork &thisNetwork, const std::vector<std::vector<double>>& featuresMatrix, const std::vector<std::vector<double>>& labels) override;

    double calculateLoss(const std::vector<double> &predictions, const std::vector<double> &labels) override;

    float getMutationRate() const
    {
        return this->mutationRate;
    }

    void setMutationRate(float mutationRate)
    {
        this->mutationRate = mutationRate;
    }

    int getPopulationSize() const
    {
        return this->populationSize;
    }

    void setPopulationSize(int populationSize)
    {
        this->populationSize = populationSize;
    }

    int getMaxGenerations() const
    {
        return this->maxGenerations;
    }

    void setMaxGenerations(int maxGenerations)
    {
        this->maxGenerations = maxGenerations;
    }

    LossFunction getLossFunction() const
    {
        return this->lossFunction;
    }

    void setLossFunction(LossFunction lossFunction)
    {
        this->lossFunction = lossFunction;
    }
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