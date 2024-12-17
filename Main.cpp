#include "ML-Models/linear_regression.cpp"
#include "ML-Models/logistic_regression.cpp"
#include <iostream>
#include "LinearAlgebra.h"
#include "Output.h"
#include <iomanip> // Required for std::fixed and std::setprecision
#include "ReadCSV.h"
#include "LinearAlgebra.h"
#include <string>
#include <cmath>


int main() {
    //Parse the CSV into a matrix
    std::vector<std::string> headers = getCSVHeaders("/Users/malikkurtz/Coding/TradingSystem/ALL CSV FILES - 2nd Edition/Smarket.csv");
    std::vector<std::vector<float>> data = parseCSV("/Users/malikkurtz/Coding/TradingSystem/ALL CSV FILES - 2nd Edition/Smarket.csv");
    std::vector<std::string> indices = getCSVIndices("/Users/malikkurtz/Coding/TradingSystem/ALL CSV FILES - 2nd Edition/Smarket.csv", 0);
    
    // std::cout << "Data Matrix Begin: " << std::endl;
    // printMatrix(data);
    // std::cout << "Data Matrix End: " << std::endl;

    // Extract the labels (last column)
    std::vector<float> labels = getColumn(data, data[0].size() - 1);

    for (int i = 0; i < labels.size(); i++) {
        if (labels[i] < 0) {
            labels[i] = 0;
        } else {
            labels[i] = 1;
        }
    }
    // std::cout << "Labels Begin: " << std::endl;
    // printVector(labels);
    // std::cout << "Labels End: " << std::endl;


    // Remove the last column from the data matrix (features)
    deleteColumn(data, data[0].size() - 1);

    

    // Initialize and fit the Linear Regression model
    LogisticRegression LR(0.01);
    LR.fit(data, labels);


    std::cout << "Coefficients Begin: " << std::endl;
    std::cout << "Intercept: " << LR.b << std::endl;
    std::cout << "Parameters" << std::endl;
    printVector(LR.parameters);
    std::cout << "Coefficients End: " << std::endl;

    // Generate predictions and compare with labels
    std::vector<float> predictions = LR.getPredictions(data);

    // std::cout << "Prediction Vs Label Begin: " << std::endl;
    // for (int i = 0; i < predictions.size(); i++) {
    //     std::cout << "(" << predictions[i] << " ," << labels[i] << ")" << std::endl;
    // }
    // std::cout << "Prediction Vs Label End: " << std::endl;

    std::cout << "Log Loss: " << LR.loss << std::endl;


    
    toCSV("results.csv", data, labels, predictions);
    return 0;

}

int main1() {

    std::vector<std::vector<float>> data = {
        {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19},  // Weight of a mouse in pounds
        {0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 1, 1, 1, 1, 1, 1},  // 0 = Not obese, 1 = Obese
    };

    data = takeTranspose(data);


    std::vector<float> labels = getColumn(data, data[0].size()-1);



    deleteColumn(data,data[0].size()-1);

    std::cout << "Original Data:" << std::endl;
    printMatrix(data);

    data = normalizeData(data);

    std::cout << "Normalized Data:" << std::endl;
    printMatrix(data);


    // printMatrix(data);
    // printVector(labels);
    LogisticRegression LR(0.01);

    LR.fit(data, labels);

    std::vector<float> predictions = LR.getPredictions(data);
    
    std::cout << predictions.size() << std::endl;

    std::cout << "Prediction Vs Label Begin: " << std::endl;
    for (int i = 0; i < predictions.size(); i++) {
        std::cout << "(" << predictions[i] << " ," << labels[i] << ")" << std::endl;
    }
    std::cout << "Prediction Vs Label End: " << std::endl;

    std::cout << "Log Loss: " << LR.loss << std::endl;

    print("Final Parameters");
    printVector(LR.parameters);
    //toCSV("results.csv", data, labels, predictions);
    return 0;


}

