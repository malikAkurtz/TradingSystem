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

bool DEBUG = false;

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
    std::vector<std::vector<double>> labels1 = {
    {5.0}, {6.0}, {7.1}, {8.0},
    {9.3}, {9.8}, {10.5}, {11.0},
    {13.2}, {14.1}, {15.0}, {15.6},
    {16.5}, {17.1}, {18.3}, {18.7},
    {19.7}, {20.3}, {21.9}, {22.5},
    {23.1}, {23.9}, {24.6}, {25.2},
    {26.0}, {26.5}, {27.1}, {27.7},
    {28.4}, {29.0}
    };


    std::vector<std::vector<double>> features2 = {
    {1}, 
    {2}, 
    {3}, {4}, {5}, {6}, {7}, {8}, {9}, {10},
    {11}, {12}, {13}, {14}, {15}, {16}, {17}, {18}, {19}, {20}
    };

    std::vector<std::vector<double>> labels2 = {
        {5},  
        {7},  
        {9},  {11}, {13}, {15}, {17}, {19}, {21}, {23},
        {25}, {27}, {29}, {31}, {33}, {35}, {37}, {39}, {41}, {43}
    };

    if (DEBUG) {
        features2 = {{2}};
        labels2 = {{7}};
    }


    std::vector<std::vector<double>> features3 = {
        {1.0, 2.5}, 
        {1.5, 3.0}, {2.0, 3.5}, {2.5, 4.0}, {3.0, 4.5},
        {3.5, 5.0}, {4.0, 5.5}, {4.5, 6.0}, {5.0, 6.5}, {5.5, 7.0},
        {6.0, 7.5}, {6.5, 8.0}, {7.0, 8.5}, {7.5, 9.0}, {8.0, 9.5},
        {8.5, 10.0}, {9.0, 10.5}, {9.5, 11.0}, {10.0, 11.5}, {10.5, 12.0}
    };

    std::vector<std::vector<double>> labels3 = {
        {5.0, 10.0}, 
        {6.0, 12.0}, {7.0, 14.0}, {8.0, 16.0}, {9.0, 18.0},
        {10.0, 20.0}, {11.0, 22.0}, {12.0, 24.0}, {13.0, 26.0}, {14.0, 28.0},
        {15.0, 30.0}, {16.0, 32.0}, {17.0, 34.0}, {18.0, 36.0}, {19.0, 38.0},
        {20.0, 40.0}, {21.0, 42.0}, {22.0, 44.0}, {23.0, 46.0}, {24.0, 48.0}
    };

    std::vector<std::vector<double>> features4 = {
        {0.0, 0.0}, // Class 0
        {0.0, 1.0}, // Class 1
        {1.0, 0.0}, // Class 1
        {1.0, 1.0}, // Class 0
        {0.5, 0.5}, // Class 0
        {0.2, 0.8}, // Class 1
        {0.8, 0.2}, // Class 1
        {0.9, 0.9}, // Class 0
        {0.3, 0.7}, // Class 1
        {0.7, 0.3}  // Class 1
    };

    std::vector<std::vector<double>> labels4 = {
        {0.0}, // Class 0
        {1.0}, // Class 1
        {1.0}, // Class 1
        {0.0}, // Class 0
        {0.0}, // Class 0
        {1.0}, // Class 1
        {1.0}, // Class 1
        {0.0}, // Class 0
        {1.0}, // Class 1
        {1.0}  // Class 1
    };




    // Wrap datasets in pairs for easy management
    std::pair<std::vector<std::vector<double>>, std::vector<std::vector<double>>> data1 = {features1, labels1};
    std::pair<std::vector<std::vector<double>>, std::vector<std::vector<double>>> data2 = {features2, labels2};
    std::pair<std::vector<std::vector<double>>, std::vector<std::vector<double>>> data3 = {features3, labels3};
    std::pair<std::vector<std::vector<double>>, std::vector<std::vector<double>>> data4 = {features4, labels4};



    // Select dataset (change this to switch datasets)
    auto& selected_data = data2; // Use data1, data2, or data3
    auto& features = selected_data.first;
    auto& labels = selected_data.second;


    if (!DEBUG) {
        features = normalizeData(features);
    }

    // Neural Network initialization
    int num_features = features[0].size();
    int num_labels = labels[0].size();  // Ensure compatibility with multiple outputs
    int num_epochs = 10000;

    if (DEBUG) {
        num_epochs = 1;
    }
    
    NeuralNetwork Network(0.01, num_epochs);  // Learning rate = 0.01, epochs = 1000
    Network.addInputLayer(std::make_shared<InputLayer>(num_features));
    // Network.addLayer(std::make_shared<Layer>(3, RELU));
    // Network.addLayer(std::make_shared<Layer>(10, RELU));
    // Network.addLayer(std::make_shared<Layer>(3, RELU));
    Network.addLayer(std::make_shared<Layer>(num_labels, NONE));  // Output layer with num_labels neurons
    

    // Train the model
    Network.fit(features, labels);

    // Evaluate the model
    std::vector<std::vector<std::vector<double>>> predictions = Network.getPredictions(features);
    std::cout << "Predictions vs Labels" << std::endl;
    printPredictionsVSLabels(predictions, labels);
    std::vector<int> epochs(num_epochs);
    std::vector<double> losses = Network.epoch_losses;
    std::iota(epochs.begin(), epochs.end(), 1); // Fills with 1 to 1000
    toCSV("training_loss.txt", epochs, Network.epoch_losses);

    std::cout << "Trained Model MSE" << std::endl;
    std::cout << Network.model_loss << std::endl;


    // Print the weights of the hidden layers
    print("Final Layer Parameters Starting from First Hidden Layer");
    for (size_t i = 0; i < Network.num_layers; i++) {
        printMatrix(Network.layers[i]->getWeightsMatrix());
    }


    return 0;
}



