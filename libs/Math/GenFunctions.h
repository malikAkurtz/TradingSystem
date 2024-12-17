#ifndef GEN_FUNCS
#define GEN_FUNCS

#include <vector>
#include <cmath>

float calculateMSE(std::vector<float> predictions, std::vector<float> labels);
float calculateLogLoss(std::vector<float> predictions, std::vector<float> labels);
std::vector<float> thresholdFunction(std::vector<float> softPredictions, float threshhold);
float sigmoid(float value);
float calculateMean(std::vector<float> v1);
float calculateSTD(std::vector<float> v1);
std::vector<std::vector<float>> normalizeData(std::vector<std::vector<float>> featuresMatrix);

#endif