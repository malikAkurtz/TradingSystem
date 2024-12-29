#ifndef GD_OPTIMIZER
#define GD_OPTIMIZER

#include "Optimizer.h"
#include "LossFunctionTypes.h"
#include "GenFunctions.h"
#include "NeuralNetwork.h"

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