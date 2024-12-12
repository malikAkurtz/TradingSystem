#ifndef UTILITY_H
#define UTILITY_H

#include <vector>
#include <cmath>

template <typename T>
void printVector(const std::vector<T>& vec) {
    std::cout << "[ ";
    for (const auto& elem : vec) {
        std::cout << elem << " ";
    }
    std::cout << "]" << std::endl;
}

template <typename T>
std::vector<std::vector<T>> takeTranspose(std::vector<std::vector<T>> inputMatrix) {
    int numRows = inputMatrix.size();
    int numCols = inputMatrix[0].size();  // Get the number of columns from the first row

    // Initialize the transposed matrix with dimensions numCols x numRows
    std::vector<std::vector<T>> transposed(numCols, std::vector<T>(numRows));

    // Fill the transposed matrix
    for (int i = 0; i < numRows; i++) {
        for (int j = 0; j < numCols; j++) {
            transposed[j][i] = inputMatrix[i][j];
        }
    }

    return transposed;
}

template <typename T>
T calculateMSE(std::vector<T> predictions, std::vector<T> labels) {
    T cumSum = 0;
    for (int i = 0; i < predictions.size(); i++) {
        for (int j = 0; j < labels.size(); j++) {
            cumSum += pow(labels[i] - predictions[i], 2);
        }
    }
    return cumSum / predictions.size();
}




#endif