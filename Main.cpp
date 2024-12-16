#include "ML-Models/linear_regression.cpp"
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
    std::vector<std::string> headers = getCSVHeaders("/Users/malikkurtz/Coding/TradingSystem/ALL CSV FILES - 2nd Edition/Boston.csv");
    std::vector<std::vector<float>> data = parseCSV("/Users/malikkurtz/Coding/TradingSystem/ALL CSV FILES - 2nd Edition/Boston.csv");
    std::vector<std::string> indices = getCSVIndices("/Users/malikkurtz/Coding/TradingSystem/ALL CSV FILES - 2nd Edition/Boston.csv", 0);
    
    
    // Extract the labels (last column)
    std::vector<float> labels = getColumn(data, data[0].size() - 1);
    

    // Remove the last column from the data matrix (features)
    deleteColumn(data, data[0].size() - 1);


    // Initialize and fit the Linear Regression model
    LinearRegression LR;
    LR.fit(data, labels);


    std::cout << "Coefficients: ";
    printVector(LR.parameters);

    // Generate predictions and compare with labels
    std::vector<float> predictions = LR.getPredictions(data);

    for (int i = 0; i < predictions.size(); i++) {
        std::cout << "(" << predictions[i] << " ," << labels[i] << ")" << std::endl;
    }

    std::cout << "Mean Squarred Error: " << LR.loss << std::endl;


    toCSV("results.csv", data, labels, predictions);
    return 0;

}

int main1() {

    std::vector<std::vector<float>> test = {
        {1,2,3},
        {1,2,3},
        {1,2,3},
        {1,2,3}
    };

    std::vector<float> col = getColumn(test, 1);


    printVector(col);


    return 0;
}

