#include "ML-Models/linear_regression.cpp"
#include "ML-Models/logistic_regression.cpp"
#include "ML-Models/NeuralNetwork.cpp"
#include <iostream>
#include "LinearAlgebra.h"
#include "Output.h"
#include <iomanip> // Required for std::fixed and std::setprecision
#include "ReadCSV.h"
#include "LinearAlgebra.h"
#include <string>
#include <cmath>
#include "Neuron.h"
#include "NetworkLayers.h"


int main() {

    std::vector<std::vector<double>> features = {
        {1.0, 2.5},// {1.5, 3.1}, {2.0, 3.7}, {2.5, 4.0},
        // {3.1, 4.2}, {3.5, 4.6}, {4.0, 5.1}, {4.5, 5.6},
        // {5.2, 6.8}, {5.8, 7.1}, {6.3, 7.5}, {6.9, 7.7},
        // {7.4, 8.0}, {7.9, 8.6}, {8.5, 9.1}, {8.8, 9.6},
        // {9.0, 10.2}, {9.5, 10.8}, {10.1, 11.5}, {10.6, 12.1},
        // {11.0, 12.8}, {11.5, 13.3}, {12.0, 14.0}, {12.5, 14.6},
        // {13.0, 15.2}, {13.5, 15.7}, {14.0, 16.3}, {14.5, 16.8},
        // {15.0, 17.4}, {15.5, 18.0}
    };

    std::vector<double> labels = {
        5.0, //6.0, 7.1, 8.0,
        // 9.3, 9.8, 10.5, 11.0,
        // 13.2, 14.1, 15.0, 15.6,
        // 16.5, 17.1, 18.3, 18.7,
        // 19.7, 20.3, 21.9, 22.5,
        // 23.1, 23.9, 24.6, 25.2,
        // 26.0, 26.5, 27.1, 27.7,
        // 28.4, 29.0
    };

    features = {
        // {1},
        {2},
        {3},
        // {4},
        // {5},
        // {6},
        // {7},
        // {8},
        // {9},
        // {10},
        // {11},
        // {12},
        // {13},
        // {14},
        // {15},
        // {16},
    };

    labels = {
        // 5,  // y = 2(1) + 3
        7,  // y = 2(2) + 3
        9,  // y = 2(3) + 3
        // 11, // y = 2(4) + 3
        // 13, // y = 2(5) + 3
        // 15, // y = 2(6) + 3
        // 17, // y = 2(7) + 3
        // 19, // y = 2(8) + 3
        // 21, // y = 2(9) + 3
        // 23, // y = 2(10) + 3
        // 25, // y = 2(11) + 3
        // 27, // y = 2(12) + 3
        // 29, // y = 2(13) + 3
        // 31, // y = 2(14) + 3
        // 33, // y = 2(15) + 3
        // 35  // y = 2(16) + 3
    };




    //features = normalizeData(features);
    int num_features = features[0].size();
    NeuralNetwork Network(0.001, 1);


    
    Network.addInputLayer(std::make_shared<InputLayer>(num_features));
    Network.addHiddenLayer(std::make_shared<HiddenLayer>(2, num_features, RELU));
    // Network.addHiddenLayer(std::make_shared<HiddenLayer>(4, 3, RELU));
    // Network.addHiddenLayer(std::make_shared<HiddenLayer>(3, 4, RELU));
    Network.addOutputLayer(std::make_shared<OutputLayer>(1, 2, NONE));

    Network.fit(features, labels);
    
    std::vector<double> predictions = Network.getPredictions(features);
    printPredictionsVSLabels(predictions, labels);
    print("Trained Model Loss");
    std::cout << Network.model_loss << std::endl;



    return 0;
}