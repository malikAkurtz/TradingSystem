#ifndef GEN_FUNCS
#define GEN_FUNCS

#include <vector>
#include <cmath>

float calculateMSE(std::vector<float> predictions, std::vector<float> labels);
float calculateLogLoss(std::vector<float> predictions, std::vector<float> labels);
std::vector<bool> thresholdFunction(std::vector<float> softPredictions, float threshhold);


#endif