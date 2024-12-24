#include "LossFunctions.h"

using namespace LinearAlgebra;

double vectorizedModifiedSquarredError(const std::vector<double> &predictions, const std::vector<double> &labels)
{
    if (predictions.size() != labels.size())
    {
        throw std::invalid_argument("Size mismatch between predictions and labels");
    }
    std::vector<double> error = subtractVectors(predictions, labels);
    return (innerProduct(error, error) / 2);
}


double vectorizedLogLoss(const std::vector<double> &predictions, const std::vector<double> &labels)
{
    if (predictions.size() != labels.size())
    {
        throw std::invalid_argument("Size mismatch between predictions and labels");
    }
    double cumSum = 0;
    const double epsilon = 1e-10;


    for (int i = 0; i < predictions.size(); i++)
    {
        double clipped_prediction = std::max(epsilon, std::min(1 - epsilon, predictions[i]));
        cumSum += (labels[i] * std::log(clipped_prediction)) + ((1 - labels[i]) * std::log((1 - clipped_prediction)));
    }

    return (-1 * (cumSum / predictions.size()));
}

