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
    {1.0, 2.5},
    // {2.0, 3.7},
    // {3.1, 4.2},
    // {4.0, 5.1},
    // {5.2, 6.8},
    // {6.3, 7.5},
    // {7.4, 8.0},
    // {8.5, 9.1},
    // {9.0, 10.2},
    // {10.1, 11.5}
};
    std::vector<double> labels = {
    5.0,
    // 7.1,
    // 9.3,
    // 10.5,
    // 13.2,
    // 15.0,
    // 16.5,
    // 18.3,
    // 19.7,
    // 21.9
};

    std::vector<std::vector<double>> features_T = takeTranspose(features);
    int num_features = features_T.size();
    
    NeuralNetwork Network;

    std::shared_ptr<InputLayer> inputLayer = std::make_shared<InputLayer>(num_features);
    std::shared_ptr<HiddenLayer> hiddenLayer1 = std::make_shared<HiddenLayer>(3, num_features, RELU);
    std::shared_ptr<OutputLayer> outputLayer = std::make_shared<OutputLayer>(1, 3, NONE);


    
    Network.addInputLayer(inputLayer);
    Network.addHiddenLayer(hiddenLayer1);
    Network.addOutputLayer(outputLayer);

    std::vector<double> predictions = Network.getPredictions(features);

    

    //Network.fit(features, labels);
    return 0;
}

