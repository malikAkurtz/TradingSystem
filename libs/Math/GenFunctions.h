#ifndef GEN_FUNCS
#define GEN_FUNCS

#include <vector>
#include <cmath>

double calculateMSE(std::vector<double> predictions, std::vector<double> labels);
double calculateLogLoss(std::vector<double> predictions, std::vector<double> labels);
std::vector<double> thresholdFunction(std::vector<double> softPredictions, double threshhold);
double sigmoid(double value);
double calculateMean(std::vector<double> v1);
double calculateSTD(std::vector<double> v1);
std::vector<std::vector<double>> normalizeData(std::vector<std::vector<double>> featuresMatrix);
double ReLU(double value);
double d_ReLU(double x);
double calculateMSE_Simple(std::vector<double> predictions, std::vector<double> labels);

#endif