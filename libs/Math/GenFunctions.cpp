#include <vector>
#include <cmath>
#include "LinearAlgebra.h"

float calculateMSE(std::vector<float> predictions, std::vector<float> labels) {
    int num_elements = predictions.size();

    std::vector<float> resultant = subtractVectors(labels, predictions);
    return innerProduct(resultant, resultant) / num_elements;
}


float calculateLogLoss(std::vector<float> predictions, std::vector<float> labels) {
    float cumSum = 0;
    for (int i = 0; i < predictions.size(); i++) {
        cumSum += (labels[i] * std::log(predictions[i])) + ((1-labels[i]) * std::log((1+1e-5)-predictions[i]));
    }
    return (-1 * (cumSum / predictions.size()));
}



std::vector<bool> thresholdFunction(std::vector<float> softPredictions, float threshhold) {
    std::vector<bool> hardPredictions(softPredictions.size());

    for (int i = 0; i < softPredictions.size(); i++) {
        if (softPredictions[i] >= threshhold) {
            hardPredictions[i] = 1;
        } else {
            hardPredictions[i] = 0;
        }
    }

    return hardPredictions;
}
