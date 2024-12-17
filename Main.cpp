#include "ML-Models/linear_regression.cpp"
#include "ML-Models/logistic_regression.cpp"
#include "ML-Models/Neuron.cpp"
#include <iostream>
#include "LinearAlgebra.h"
#include "Output.h"
#include <iomanip> // Required for std::fixed and std::setprecision
#include "ReadCSV.h"
#include "LinearAlgebra.h"
#include <string>
#include <cmath>


int main1() {
    //Parse the CSV into a matrix
    std::vector<std::string> headers = getCSVHeaders("/Users/malikkurtz/Coding/TradingSystem/ALL CSV FILES - 2nd Edition/Smarket.csv");
    std::vector<std::vector<double>> data = parseCSV("/Users/malikkurtz/Coding/TradingSystem/ALL CSV FILES - 2nd Edition/Smarket.csv");
    std::vector<std::string> indices = getCSVIndices("/Users/malikkurtz/Coding/TradingSystem/ALL CSV FILES - 2nd Edition/Smarket.csv", 0);
    
    // std::cout << "Data Matrix Begin: " << std::endl;
    // printMatrix(data);
    // std::cout << "Data Matrix End: " << std::endl;

    // Extract the labels (last column)
    std::vector<double> labels = getColumn(data, data[0].size() - 1);

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
    deleteColumn(data, data[0].size() - 1);
    deleteColumn(data, data[0].size() - 1);
    deleteColumn(data, data[0].size() - 1);
    deleteColumn(data, data[0].size() - 1);
    data = normalizeData(data);
    // Initialize and fit the Linear Regression model
    LogisticRegression LR(0.01);
    LR.fit(data, labels);


    std::cout << "Coefficients Begin: " << std::endl;
    std::cout << "Parameters" << std::endl;
    printVector(LR.parameters);
    std::cout << "Coefficients End: " << std::endl;

    // Generate predictions and compare with labels
    std::vector<double> predictions = LR.getPredictions(data);

    // std::cout << "Prediction Vs Label Begin: " << std::endl;
    // for (int i = 0; i < predictions.size(); i++) {
    //     std::cout << "(" << predictions[i] << " ," << labels[i] << ")" << std::endl;
    // }
    // std::cout << "Prediction Vs Label End: " << std::endl;

    std::cout << "Log Loss: " << LR.loss << std::endl;

    toCSV("results.csv", data, labels, predictions);
    return 0;

}

int main2() {

    std::vector<std::vector<double>> data = {
        {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19},  // Weight of a mouse in pounds
        {0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 1, 1, 1, 1, 1, 1},  // 0 = Not obese, 1 = Obese
    };

    data = takeTranspose(data);


    std::vector<double> labels = getColumn(data, data[0].size()-1);



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

    std::vector<double> predictions = LR.getPredictions(data);
    
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

int main() {

    std::vector<std::vector<double>> features = {
    {1.0, 2.5},
    {2.0, 3.7},
    {3.1, 4.2},
    {4.0, 5.1},
    {5.2, 6.8},
    {6.3, 7.5},
    {7.4, 8.0},
    {8.5, 9.1},
    {9.0, 10.2},
    {10.1, 11.5}
};
    std::vector<double> labels = {
    5.0,
    7.1,
    9.3,
    10.5,
    13.2,
    15.0,
    16.5,
    18.3,
    19.7,
    21.9
};

    std::vector<std::vector<double>> features_T = takeTranspose(features);

    NeuralNetwork Network;
    NetworkLayer inputLayer;
    NetworkLayer hiddenLayer1;
    NetworkLayer outputLayer;
    Network.layers.push_back(inputLayer);
    Network.layers.push_back(hiddenLayer1);
    Network.layers.push_back(outputLayer);
    for (int i = 0; i < features_T.size(); i++) {
        Neuron neuron(features_T[i]);
        inputLayer.nodes.push_back(neuron);
    }
    


    return 0;
}

