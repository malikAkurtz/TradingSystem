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

bool DEBUG = true;


int main() {
    // Define datasets
    std::vector<std::vector<double>> features1 = {
        {1.0, 2.5}, {1.5, 3.1}, {2.0, 3.7}, {2.5, 4.0},
        {3.1, 4.2}, {3.5, 4.6}, {4.0, 5.1}, {4.5, 5.6},
        {5.2, 6.8}, {5.8, 7.1}, {6.3, 7.5}, {6.9, 7.7},
        {7.4, 8.0}, {7.9, 8.6}, {8.5, 9.1}, {8.8, 9.6},
        {9.0, 10.2}, {9.5, 10.8}, {10.1, 11.5}, {10.6, 12.1},
        {11.0, 12.8}, {11.5, 13.3}, {12.0, 14.0}, {12.5, 14.6},
        {13.0, 15.2}, {13.5, 15.7}, {14.0, 16.3}, {14.5, 16.8},
        {15.0, 17.4}, {15.5, 18.0}
    };
    std::vector<double> labels1 = {
        5.0, 6.0, 7.1, 8.0,
        9.3, 9.8, 10.5, 11.0,
        13.2, 14.1, 15.0, 15.6,
        16.5, 17.1, 18.3, 18.7,
        19.7, 20.3, 21.9, 22.5,
        23.1, 23.9, 24.6, 25.2,
        26.0, 26.5, 27.1, 27.7,
        28.4, 29.0
    };

    std::vector<std::vector<double>> features2 = {
        {1}, {2}, {3}, {4}, {5}, {6}, {7}, {8}, {9}, {10},
        {11}, {12}, {13}, {14}, {15}, {16}
    };
    std::vector<double> labels2 = {
        5, 7, 9, 11, 13, 15, 17, 19, 21, 23, 25, 27, 29, 31, 33, 35
    };

    std::vector<std::vector<double>> features3 = {
    {1.0, 2.5}, //{1.5, 3.1}, {2.0, 3.7}, {2.5, 4.0},
    // {3.1, 4.2}, {3.5, 4.6}, {4.0, 5.1}, {4.5, 5.6},
    // {5.2, 6.8}, {5.8, 7.1}, {6.3, 7.5}, {6.9, 7.7},
    // {7.4, 8.0}, {7.9, 8.6}, {8.5, 9.1}, {8.8, 9.6},
    // {9.0, 10.2}, {9.5, 10.8}, {10.1, 11.5}, {10.6, 12.1},
    // {11.0, 12.8}, {11.5, 13.3}, {12.0, 14.0}, {12.5, 14.6},
    // {13.0, 15.2}, {13.5, 15.7}, {14.0, 16.3}, {14.5, 16.8},
    // {15.0, 17.4}, {15.5, 18.0}
};

std::vector<std::vector<double>> labels3 = {
    {5.0, 10.0}, //{6.0, 12.0}, {7.1, 14.2}, {8.0, 16.0},
    // {9.3, 18.6}, {9.8, 19.6}, {10.5, 21.0}, {11.0, 22.0},
    // {13.2, 26.4}, {14.1, 28.2}, {15.0, 30.0}, {15.6, 31.2},
    // {16.5, 33.0}, {17.1, 34.2}, {18.3, 36.6}, {18.7, 37.4},
    // {19.7, 39.4}, {20.3, 40.6}, {21.9, 43.8}, {22.5, 45.0},
    // {23.1, 46.2}, {23.9, 47.8}, {24.6, 49.2}, {25.2, 50.4},
    // {26.0, 52.0}, {26.5, 53.0}, {27.1, 54.2}, {27.7, 55.4},
    // {28.4, 56.8}, {29.0, 58.0}
};

    // Wrap datasets in pairs for easy management
    std::pair<std::vector<std::vector<double>>, std::vector<double>> data1 = {features1, labels1};
    std::pair<std::vector<std::vector<double>>, std::vector<double>> data2 = {features2, labels2};
    std::pair<std::vector<std::vector<double>>, std::vector<std::vector<double>>> data3 = {features3, labels3};

    // Select dataset (change this to switch datasets)
    auto& selected_data = data2; // Use data1 or data2
    auto& features = selected_data.first;
    auto& labels = selected_data.second;

    // Normalize features if required
    //features = normalizeData(features); // UNCOMMENT THIS

    // Neural Network initialization
    int num_features = features[0].size();
    //int num_labels = labels.size();
    NeuralNetwork Network(0.01, 1000);
    Network.addInputLayer(std::make_shared<InputLayer>(num_features));
    // Network.addHiddenLayer(std::make_shared<HiddenLayer>(4, num_features, RELU));
    Network.addOutputLayer(std::make_shared<OutputLayer>(1, num_features, NONE)); // 1 or labels[0].size() depending on dataset

    // Train the model
    Network.fit(features, labels);

    // Evaluate the model
    std::vector<double> predictions = Network.getPredictions(features);
    std::cout << "Predictions vs Labels" << std::endl;
    printPredictionsVSLabels(predictions, labels);

    std::cout << "Trained Model MSE" << std::endl;
    std::cout << Network.model_loss << std::endl;

    print("Hidden Layer Parameters Starting from First Hidden Layer");
    for (int i = 0; i < Network.hiddenLayers.size(); i++) {
        printMatrix_(Network.hiddenLayers[i]->getWeightsMatrix());
    }
    print("Trained Model Output Layer Parameters");
    printMatrix_(Network.outputLayer->getWeightsMatrix());

    std::vector<double> vector = {1,2,3};
    printMatrix(vector1DtoColumnVector(vector));

    return 0;
}
