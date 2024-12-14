#include <vector>
#include <cmath>

float calculateMSE(std::vector<float> predictions, std::vector<float> labels) {
    float cumSum = 0;
    for (int i = 0; i < predictions.size(); i++) {
        // change this to be a matrix multiply
        cumSum += pow(labels[i] - predictions[i], 2);
    }
    return cumSum / predictions.size();
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
