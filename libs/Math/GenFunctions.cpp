#include <vector>
#include <cmath>
#include "LinearAlgebra.h"
#include <iostream>
#include "Output.h"

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

float sigmoid(float value) {
    if (value >= 0) {
        return 1.0 / (1.0 + std::exp(-value));
    } else {
        float exp_val = std::exp(value);
        return exp_val / (1.0 + exp_val);
    }
}

float calculateMean(std::vector<float> v1) {
    return (accumulateVector(v1) / v1.size());
}

float calculateSTD(std::vector<float> v1) {
    int num_elements = v1.size();
    float mean = calculateMean(v1);

    std::vector<float> mean_vector = createVector(mean, num_elements);

    std::vector<float> normalized = subtractVectors(v1, mean_vector);
    float IP = innerProduct(normalized, normalized);

    return (std::sqrt(IP / num_elements));

}

std::vector<std::vector<float>> normalizeData(std::vector<std::vector<float>> featuresMatrix) {
    int num_cols = featuresMatrix[0].size();
    int num_rows = featuresMatrix.size();
    std::vector<std::vector<float>> normalized_matrix(num_rows, std::vector<float>(num_cols, 0));

    // for every column
    for (int j = 0; j < num_cols; j++) {
        std::vector<float> col_to_normalize = getColumn(featuresMatrix, j);
        std::vector<float> pre_normalized = subtractVectors(col_to_normalize, createVector(calculateMean(col_to_normalize), num_rows));
        float col_STD = calculateSTD(col_to_normalize);
        if (col_STD == 0) {
            throw std::runtime_error("Standard deviation is zero, cannot normalize.");
        }
        std::vector<float> normalized = divideVector(pre_normalized, col_STD);

        updateColumn(normalized_matrix, normalized, j);
    }

    return normalized_matrix;
}

