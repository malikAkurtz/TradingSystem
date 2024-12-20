#include <iostream>
#include <vector>
#include <string>
#include "Output.h"

void printVector(const std::vector<double>& vec) {
    if (DEBUG) {
        std::cout << "< ";
        for (const auto& elem : vec) {
            std::cout << elem << " ";
        }
        std::cout << ">" << std::endl;
    }
}

void printMatrix(const std::vector<std::vector<double>>& matrix) {
    if (DEBUG) {
        std::cout << "[" << std::endl;
        for (const auto& row : matrix) {
            std::cout << "  < ";
            for (const auto& elem : row) {
                std::cout << elem << " ";
            }
            std::cout << ">" << std::endl;
        }
        std::cout << "]" << std::endl;
    }
}

void print(std::string toPrint) {
    if (DEBUG) {
        std::cout << toPrint << std::endl;
    }
}

void printPredictionsVSLabels(const std::vector<std::vector<std::vector<double>>>& predictions, 
                              const std::vector<std::vector<double>>& labels) {
    if (predictions.size() != labels.size()) {
        std::cerr << "Error: Predictions and labels sizes do not match!" << std::endl;
        return;
    }

    // Iterate over all prediction-label pairs
    for (size_t i = 0; i < predictions.size(); i++) {
        std::cout << "Prediction-Label Pair " << i + 1 << ": ";
        std::cout << "<";

        // Flatten the 2D column vector in `predictions[i]` and compare with the corresponding `labels[i]`
        if (predictions[i].size() != labels[i].size()) {
            std::cerr << "Error: Mismatch in the number of outputs for prediction and label at index " << i << "!" << std::endl;
            continue;
        }

        for (size_t j = 0; j < predictions[i].size(); j++) {
            // Extract the single value from the column vector (always the first column)
            std::cout << "<" << predictions[i][j][0] << ", " << labels[i][j] << ">";
        }

        std::cout << ">" << std::endl;
    }
}


void printMatrixShape(std::vector<std::vector<double>> matrix) {
    if (DEBUG) {std::cout << "(" << matrix.size() << "," << matrix[0].size() << ")" << std::endl;}
}

void printVectorShape(std::vector<double> vector) {
    if (DEBUG) {std::cout << "(" << vector.size() << ", " << ")" << std::endl;}
}

void printMatrix_(const std::vector<std::vector<double>>& matrix) {
    std::cout << "[" << std::endl;
    for (const auto& row : matrix) {
        std::cout << "  < ";
        for (const auto& elem : row) {
            std::cout << elem << " ";
        }
        std::cout << ">" << std::endl;
    }
    std::cout << "]" << std::endl;
}