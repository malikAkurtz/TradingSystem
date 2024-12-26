#ifndef ACTIVATION_FUNCS
#define ACTIVATION_FUNCS

#include <vector>

namespace ActivationFunctions
{
    std::vector<std::vector<double>> matrix_ReLU(const std::vector<std::vector<double>> &v1);
    std::vector<std::vector<double>> matrix_d_ReLU(const std::vector<std::vector<double>> &v1);
    std::vector<std::vector<double>> matrix_sigmoid(const std::vector<std::vector<double>> &v1);
    double sigmoid_single(const double &value);
    std::vector<std::vector<double>> matrix_d_sigmoid(const std::vector<std::vector<double>> &v1);

}

#endif