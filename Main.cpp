#include "ML-Models/linear_regression.cpp"
#include <iostream>
#include "LinearAlgebra.h"
#include "Output.h"
#include <iomanip> // Required for std::fixed and std::setprecision
#include "ReadCSV.h"
#include "LinearAlgebra.h"

#include <cmath>




int main() {
    //Parse the CSV into a matrix
    std::vector<std::vector<float>> data_matrix = parseCSV("/Users/malikkurtz/Coding/TradingSystem/ALL CSV FILES - 2nd Edition/Boston.csv");

    // Extract the labels (last column)
    std::vector<float> labels;
    for (const auto& row : data_matrix) {
        if (!row.empty()) {
            labels.push_back(row.back());
        }
    }

    // Remove the last column from the data matrix (features)
    for (auto& row : data_matrix) {
        row.pop_back();
    }

    // std::vector<std::vector<float>> A = {
    //     {1},
    //     {3},
    //     {4}
    // };

    // std::vector<float> b = {
    //     3,
    //     7,
    //     9
    // };



    // Initialize and fit the Linear Regression model
    LinearRegression LR;
    LR.fit(data_matrix, labels);



    std::cout << "Coefficients: ";
    printVector(LR.parameters);

    // Generate predictions and compare with labels
    std::vector<float> predictions = LR.getPredictions(data_matrix);

    for (int i = 0; i < predictions.size(); i++) {
        std::cout << "(" << predictions[i] << " ," << labels[i] << ")" << std::endl;
    }

    std::cout << "Mean Squarred Error: " << LR.loss << std::endl;
    return 0;

}
