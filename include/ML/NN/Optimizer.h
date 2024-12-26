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

    float getMutationRate() const;

    void setMutationRate(float mutationRate);

    int getPopulationSize() const;

    void setPopulationSize(int populationSize);

    int getMaxGenerations() const;

    void setMaxGenerations(int maxGenerations);

    LossFunction getLossFunction() const;

    void setLossFunction(LossFunction lossFunction);
};

class GradientDescentOptimizer : public Optimizer
{
private:
    float learningRate;
    int numEpochs;
    int batchSize;
    LossFunction lossFunction;


public:
    std::vector<double> epochLosses;
    std::vector<double> gradientNorms;

    GradientDescentOptimizer();

    GradientDescentOptimizer(float learningRate, int numEpochs, int batchSize, LossFunction lossFunction);

    void fit(NeuralNetwork &thisNetwork, const std::vector<std::vector<double>>& featuresMatrix, const std::vector<std::vector<double>>& labels) override;

    double calculateLoss(const std::vector<double> &predictions, const std::vector<double> &labels) override;

    float getLearningRate() const;

    void setLearningRate(float learningRate);

    int getNumEpochs() const;

    void setNumEpochs(int numEpochs);

    int getBatchSize() const;

    void setBatchSize(int batchSize);

    LossFunction getLossFunction() const;

    void setLossFunction(LossFunction lossFunction);
};


#endif