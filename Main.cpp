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

int main1() {
    // Define datasets
    std::pair<std::vector<std::vector<double>>, std::vector<std::vector<double>>> data1 = {{
        {1.0, 2.5}, {1.5, 3.1}, {2.0, 3.7}, {2.5, 4.0},
        {3.1, 4.2}, {3.5, 4.6}, {4.0, 5.1}, {4.5, 5.6},
        {5.2, 6.8}, {5.8, 7.1}, {6.3, 7.5}, {6.9, 7.7},
        {7.4, 8.0}, {7.9, 8.6}, {8.5, 9.1}, {8.8, 9.6},
        {9.0, 10.2}, {9.5, 10.8}, {10.1, 11.5}, {10.6, 12.1},
        {11.0, 12.8}, {11.5, 13.3}, {12.0, 14.0}, {12.5, 14.6},
        {13.0, 15.2}, {13.5, 15.7}, {14.0, 16.3}, {14.5, 16.8},
        {15.0, 17.4}, {15.5, 18.0}
    },
    {
    {5.0}, {6.0}, {7.1}, {8.0},
    {9.3}, {9.8}, {10.5}, {11.0},
    {13.2}, {14.1}, {15.0}, {15.6},
    {16.5}, {17.1}, {18.3}, {18.7},
    {19.7}, {20.3}, {21.9}, {22.5},
    {23.1}, {23.9}, {24.6}, {25.2},
    {26.0}, {26.5}, {27.1}, {27.7},
    {28.4}, {29.0}
    }};


    std::pair<std::vector<std::vector<double>>, std::vector<std::vector<double>>> data2 = {{
    {1}, {2}, {3}, {4}, {5}, {6}, {7}, {8}, {9}, {10},
    {11}, {12}, {13}, {14}, {15}, {16}, {17}, {18}, {19}, {20}
    },
    {
        {5}, {7}, {9},  {11}, {13}, {15}, {17}, {19}, {21}, {23},
        {25}, {27}, {29}, {31}, {33}, {35}, {37}, {39}, {41}, {43}
    }};

    std::pair<std::vector<std::vector<double>>, std::vector<std::vector<double>>> data3 = {{
        {1.0, 2.5}, 
        {1.5, 3.0}, {2.0, 3.5}, {2.5, 4.0}, {3.0, 4.5},
        {3.5, 5.0}, {4.0, 5.5}, {4.5, 6.0}, {5.0, 6.5}, {5.5, 7.0},
        {6.0, 7.5}, {6.5, 8.0}, {7.0, 8.5}, {7.5, 9.0}, {8.0, 9.5},
        {8.5, 10.0}, {9.0, 10.5}, {9.5, 11.0}, {10.0, 11.5}, {10.5, 12.0}
    }, {
        {5.0, 10.0}, 
        {6.0, 12.0}, {7.0, 14.0}, {8.0, 16.0}, {9.0, 18.0},
        {10.0, 20.0}, {11.0, 22.0}, {12.0, 24.0}, {13.0, 26.0}, {14.0, 28.0},
        {15.0, 30.0}, {16.0, 32.0}, {17.0, 34.0}, {18.0, 36.0}, {19.0, 38.0},
        {20.0, 40.0}, {21.0, 42.0}, {22.0, 44.0}, {23.0, 46.0}, {24.0, 48.0}
    }};

    std::pair<std::vector<std::vector<double>>, std::vector<std::vector<double>>> data4 = {{
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
    },
    {
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
    }};

    std::pair<std::vector<std::vector<double>>, std::vector<std::vector<double>>> data5 = {{
        {0.1, 0.2}, // Class 0
        {0.3, 0.7}, // Class 1
        {0.4, 0.4}, // Class 0
        {0.6, 0.8}, // Class 1
        {0.9, 0.9}, // Class 0
        {0.2, 0.3}, // Class 0
        {0.7, 0.1}, // Class 1
        {0.8, 0.6}, // Class 1
        {0.5, 0.2}, // Class 0
        {0.3, 0.9}, // Class 1
        {0.4, 0.1}, // Class 0
        {0.8, 0.4}, // Class 1
        {0.7, 0.9}, // Class 1
        {0.2, 0.8}, // Class 1
        {0.5, 0.5}, // Class 0
        {0.1, 0.9}, // Class 1
        {0.9, 0.3}, // Class 0
        {0.6, 0.2}, // Class 0
        {0.7, 0.8}, // Class 1
        {0.2, 0.6}  // Class 1
    },
    {
        {0.0}, // Class 0
        {1.0}, // Class 1
        {0.0}, // Class 0
        {1.0}, // Class 1
        {0.0}, // Class 0
        {0.0}, // Class 0
        {1.0}, // Class 1
        {1.0}, // Class 1
        {0.0}, // Class 0
        {1.0}, // Class 1
        {0.0}, // Class 0
        {1.0}, // Class 1
        {1.0}, // Class 1
        {1.0}, // Class 1
        {0.0}, // Class 0
        {1.0}, // Class 1
        {0.0}, // Class 0
        {0.0}, // Class 0
        {1.0}, // Class 1
        {1.0}  // Class 1
    }};


    std::pair<std::vector<std::vector<double>>, std::vector<std::vector<double>>> data_test =
    {{{2}, {3}}, {{7}, {9}}};


    // Select dataset (change this to switch datasets)
    auto& selected_data = data2; // Use data1, data2, or data3
    auto& features = selected_data.first;
    auto& labels = selected_data.second;


    features = normalizeData(features);

    // Neural Network initialization
    int num_features = features[0].size();
    int num_labels = labels[0].size();  // Ensure compatibility with multiple outputs
    int num_epochs = 10000;
    
    NeuralNetwork Network(0.001, num_epochs, SQUARRED_ERROR); 
    Network.addInputLayer(std::make_shared<InputLayer>(num_features));
    Network.addLayer(std::make_shared<Layer>(2, RELU, RANDOM));
    Network.addLayer(std::make_shared<Layer>(3, RELU, RANDOM));
    Network.addLayer(std::make_shared<Layer>(2, SIGMOID, RANDOM));
    Network.addLayer(std::make_shared<Layer>(num_labels, RELU, RANDOM));

    // Train the model
    Network.fit(features, labels);

    // Evaluate the model
    std::vector<std::vector<std::vector<double>>> predictions = Network.getPredictions(features);
    std::cout << "Predictions vs Labels" << std::endl;
    printPredictionsVSLabels(predictions, labels);
    std::vector<int> epochs(num_epochs);
    std::vector<double> losses = Network.epoch_losses;
    std::iota(epochs.begin(), epochs.end(), 1); // Fills with 1 to 1000
    toCSV("training_loss.txt", epochs, Network.epoch_losses, Network.epoch_gradient_norms);

    std::cout << "Trained Model Loss" << std::endl;
    std::cout << Network.model_loss << std::endl;


    // Print the weights of the hidden layers
    printDebug("Final Layer Parameters Starting from First Hidden Layer");
    for (size_t i = 0; i < Network.num_hidden_layers; i++) {
        printMatrix(Network.layers[i]->getWeightsMatrix());
    }


    return 0;
}



int main() {
    std::vector<std::vector<double>>  n = {
        {38, 50},
        {38, 50}
    };
    std::vector<std::vector<double>> m = {
        {2, 1},
        {3, 1}
    };
    std::vector<std::vector<double>> result = matrixMultiply(n, m);
    printMatrix(result);
    return 0;
}