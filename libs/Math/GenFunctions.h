#ifndef GEN_FUNCS
#define GEN_FUNCS

#include "LinearAlgebra.h"

double calculateMSE(const std::vector<double>& predictions, const std::vector<double>& labels);
double calculateLogLoss(const std::vector<double>& predictions, const std::vector<double>& labels);
std::vector<double> thresholdFunction(const std::vector<double> &softPredictions, const double &threshhold);
double calculateMean(const std::vector<double>& v1);
double calculateSTD(const std::vector<double>& v1);
std::vector<std::vector<double>> normalizeData(const std::vector<std::vector<double>>& dataMatrix);
double calculateMSE_Simple(const std::vector<double>& predictions, const std::vector<double>& labels);
std::vector<std::vector<std::vector<double>>> createBatches(const std::vector<std::vector<double>> &features, int batchSize);

#endif