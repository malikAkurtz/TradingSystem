#include <vector>
#include <cmath>
#include "LinearAlgebra.h"
#include <iostream>
#include "Output.h"

double calculateMSE(const std::vector<double>& predictions, std::vector<double>& labels) {
    int num_elements = predictions.size();

    std::vector<double> resultant = subtractVectors(predictions, labels);
    return innerProduct(resultant, resultant) / num_elements;
}

double calculateMSE_Simple(const std::vector<double>& predictions, const std::vector<double>& labels) {
    std::vector<double> resultant = subtractVectors(predictions, labels);
    return innerProduct(resultant, resultant) / 2;
}


double calculateLogLoss(const std::vector<double>& predictions, const std::vector<double>& labels) {
    double cumSum = 0;
    const double epsilon = 1e-10;
    for (int i = 0; i < predictions.size(); i++) {
        double clipped_prediction = std::max(epsilon, std::min(1-epsilon, predictions[i]));
        cumSum += (labels[i] * std::log(clipped_prediction)) + ((1-labels[i]) * std::log((1-clipped_prediction)));
    }
    return (-1 * (cumSum / predictions.size()));
}

// takes two column vectors
double modifiedSquarredError(const std::vector<std::vector<double>>& predictions, const std::vector<std::vector<double>>&  labels) {
    std::vector<double> preds1D = columnVectortoVector1D(predictions);
    std::vector<double> actual1D = columnVectortoVector1D(labels);

    if (predictions.size() != labels.size()) {
        throw std::invalid_argument("Size mismatch between predictions and labels");
    }
    std::vector<double> error = subtractVectors(preds1D, actual1D);
    return (innerProduct(error, error) / 2);
}



std::vector<double> thresholdFunction(const std::vector<double>& softPredictions, const double& threshhold) {
    std::vector<double> hardPredictions(softPredictions.size());

    for (int i = 0; i < softPredictions.size(); i++) {
        if (softPredictions[i] >= threshhold) {
            hardPredictions[i] = 1.0;
        } else {
            hardPredictions[i] = 0.0;
        }
    }

    return hardPredictions;
}

std::vector<std::vector<double>> sigmoid(const std::vector<std::vector<double>>& v1) {
    std::vector<std::vector<double>> resultant = v1;
    for (int i = 0; i < resultant.size(); i++) {
        if (resultant[i][0] >= 0) {
            resultant[i][0] = 1.0 / (1.0 + std::exp(-resultant[i][0]));
        } else {
            double exp_val = std::exp(resultant[i][0]);
            resultant[i][0] = exp_val / (1.0 + exp_val);
        }
    }
    return resultant;
}

double sigmoid_single(const double& value) {
    if (value >= 0) {
        return 1.0 / (1.0 + std::exp(-value));
    } else {
        double exp_val = std::exp(value);
        return exp_val / (1.0 + exp_val);
    }
}

std::vector<std::vector<double>> d_sigmoid(const std::vector<std::vector<double>>& v1) {
    std::vector<std::vector<double>> resultant(v1.size(), std::vector<double>(1));
    for (int i = 0; i < resultant.size(); i++) {
        double sig = sigmoid_single(v1[i][0]);
        resultant[i][0] = sig * (1-sig);
    }
    return resultant;
}

double calculateMean(const std::vector<double>& v1) {
    return (accumulateVector(v1) / v1.size());
}

double calculateSTD(const std::vector<double>& v1) {
    int num_elements = v1.size();
    double mean = calculateMean(v1);

    std::vector<double> mean_vector = createVector(mean, num_elements);

    std::vector<double> normalized = subtractVectors(v1, mean_vector);
    double IP = innerProduct(normalized, normalized);

    return (std::sqrt(IP / num_elements));

}

std::vector<std::vector<double>> normalizeData(const std::vector<std::vector<double>>& dataMatrix) {
    int num_cols = dataMatrix[0].size();
    int num_rows = dataMatrix.size();
    std::vector<std::vector<double>> normalized_matrix(num_rows, std::vector<double>(num_cols, 0));

    // for every column
    for (int j = 0; j < num_cols; j++) {
        std::vector<double> col_to_normalize = getColumn(dataMatrix, j);
        std::vector<double> pre_normalized = subtractVectors(col_to_normalize, createVector(calculateMean(col_to_normalize), num_rows));
        double col_STD = calculateSTD(col_to_normalize);
        if (col_STD == 0) {
            throw std::runtime_error("Standard deviation is zero, cannot normalize.");
        }
        std::vector<double> normalized = divideVector(pre_normalized, col_STD);

        updateColumn(normalized_matrix, normalized, j);
    }

    return normalized_matrix;
}


// takes in a column vector
std::vector<std::vector<double>> ReLU(const std::vector<std::vector<double>>& v1) {
    std::vector<std::vector<double>> resultant = v1;

    for (int i = 0; i < v1.size(); i++) {
        if (v1[i][0] < 0) {
            resultant[i][0] = 0;
        } else {
        }
    }
    return resultant;
}

std::vector<std::vector<double>> d_ReLU(const std::vector<std::vector<double>>& v1) {
    std::vector<std::vector<double>> resultant(v1.size(), std::vector<double>(1));
    for (int i = 0; i < v1.size(); i++) {
        if (v1[i][0] >= 0) {
            resultant[i][0] = 1;
        } else {
            resultant[i][0] = 0;
        }
    }
    return resultant;
}



