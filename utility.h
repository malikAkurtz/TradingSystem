#ifndef UTILITY_H
#define UTILITY_H

#include <vector>
#include <cmath>
#include <string>
#include <fstream>
#include <sstream>
#include <iostream>


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

template <typename T>
T calculateLogLoss(std::vector<T> predictions, std::vector<T> labels) {
    T cumSum = 0;
    for (int i = 0; i < predictions.size(); i++) {
        cumSum += (labels[i] * std::log(predictions[i])) + ((1-labels[i]) * std::log((1+1e-5)-predictions[i]));
    }
    return (-1 * (cumSum / predictions.size()));
}

template <typename T>
T calculateDistance(std::vector<T> v1, std::vector<T> v2) {
    std::vector<T> resultant;

    for (int i = 0; i < v1.size(); i++) {
        resultant.push_back(v2[i]-v1[i]);
    }

    double sum_of_squares = 0;
    for (int i = 0; i < resultant.size(); i++) {
        
        sum_of_squares += pow(resultant[i], 2);
    }

    return sqrt(sum_of_squares);
}

template <typename T>
std::vector<bool> thresholdFunction(std::vector<T> softPredictions, float threshhold) {
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


std::vector<std::vector<float>> parseCSV(std::string file_name) {
    std::vector<std::vector<float>> dataMatrix;

    std::ifstream file(file_name);

    std::string line;

    std::getline(file, line); // skip headers

    while (std::getline(file, line)) {
        if (line.empty()) {continue;}

        std::stringstream ss(line);

        std::string cell;
        std::vector<float> rowValues;

        while (std::getline(ss, cell, ',')) {
            if (cell == "M") {
                rowValues.push_back(1.0);
            }
            else if (cell == "B") {
                rowValues.push_back(0.0);
            } else {
                rowValues.push_back(std::stof(cell));
            }
            
        }
        dataMatrix.push_back(rowValues);
    }

    file.close();
    return dataMatrix;
}





#endif