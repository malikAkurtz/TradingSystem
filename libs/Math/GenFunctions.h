#ifndef GEN_FUNCS
#define GEN_FUNCS

#include <vector>
#include <cmath>

double calculateMSE(const std::vector<double>& predictions, const std::vector<double>& labels);
double calculateLogLoss(const std::vector<double>& predictions, const std::vector<double>& labels);
std::vector<double> thresholdFunction(const std::vector<double>& softPredictions, const double& threshhold);
std::vector<std::vector<double>> sigmoid(const std::vector<std::vector<double>>& v1); // derivative not yet implemented
double sigmoid_single(const double& value);
double calculateMean(const std::vector<double>& v1);
double calculateSTD(const std::vector<double>& v1);
std::vector<std::vector<double>> normalizeData(const std::vector<std::vector<double>>& dataMatrix);
std::vector<std::vector<double>> ReLU(const std::vector<std::vector<double>>& v1);
std::vector<std::vector<double>> d_ReLU(const std::vector<std::vector<double>>& v1);
double calculateMSE_Simple(const std::vector<double>& predictions, const std::vector<double>& labels);
double modifiedSquarredError(const std::vector<std::vector<double>>& predictions, const std::vector<std::vector<double>>& labels);
std::vector<std::vector<double>> d_sigmoid(const std::vector<std::vector<double>>& v1);

#endif