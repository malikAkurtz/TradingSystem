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
    const float epsilon = 1e-10;
    for (int i = 0; i < predictions.size(); i++) {
        float clipped_prediction = std::max(epsilon, std::min(1-epsilon, predictions[i]));
        cumSum += (labels[i] * std::log(clipped_prediction)) + ((1-labels[i]) * std::log((1-clipped_prediction)));
    }
    return (-1 * (cumSum / predictions.size()));
}



std::vector<float> thresholdFunction(std::vector<float> softPredictions, float threshhold) {
    std::vector<float> hardPredictions(softPredictions.size());

    for (int i = 0; i < softPredictions.size(); i++) {
        if (softPredictions[i] >= threshhold) {
            hardPredictions[i] = 1.0;
        } else {
            hardPredictions[i] = 0.0;
        }
    }

    return hardPredictions;
}
