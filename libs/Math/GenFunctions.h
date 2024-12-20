#ifndef GEN_FUNCS
#define GEN_FUNCS

#include <vector>
#include <cmath>

double calculateMSE(std::vector<double> predictions, std::vector<double> labels);
double calculateLogLoss(std::vector<double> predictions, std::vector<double> labels);
std::vector<double> thresholdFunction(std::vector<double> softPredictions, double threshhold);
std::vector<std::vector<double>> sigmoid(std::vector<std::vector<double>> v1); // derivative not yet implemented
double sigmoid_single(double value);
double calculateMean(std::vector<double> v1);
double calculateSTD(std::vector<double> v1);
std::vector<std::vector<double>> normalizeData(std::vector<std::vector<double>> featuresMatrix);
std::vector<std::vector<double>> ReLU(std::vector<std::vector<double>> v1);
std::vector<std::vector<double>> d_ReLU(std::vector<std::vector<double>> v1);
double calculateMSE_Simple(std::vector<double> predictions, std::vector<double> labels);
double modifiedSquarredError(std::vector<std::vector<double>> predictions, std::vector<std::vector<double>>  labels);
std::vector<std::vector<double>> d_sigmoid(std::vector<std::vector<double>> v1);

#endif