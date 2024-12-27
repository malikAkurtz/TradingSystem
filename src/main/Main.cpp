#include "NeuralNetwork.h"
#include <numeric>
#include "ReadCSV.h"
#include "BestEncoding.h"
#include "Optimizer.h"
#include "GradientDescentOptimizer.h"
#include "NeuroEvolutionOptimizer.h"
#include "TestData.h"
#include "GenFunctions.h"

using namespace LinearAlgebra;

bool DEBUG = false;

void evaluateModel(NeuralNetwork& network, const std::vector<std::vector<double>>& X_val, const std::vector<std::vector<double>>& Y_val) {
    auto predictions = network.feedForward(X_val);
    printPredictionsVSLabels(predictions, Y_val);
    std::cout << "Trained Model Loss: " << network.model_loss << std::endl;
    print("Final Layer Parameters Starting from First Hidden Layer");
    for (size_t i = 0; i < network.num_hidden_layers; i++) {
        printMatrix(network.layers[i].getWeightsMatrix());
    }
}


int main() {
    // Seed for random
    srand(static_cast<unsigned int>(time(0)));

    // Parse CSV and specify label column
    std::vector<std::vector<double>> data = parseCSV("/Users/malikkurtz/Coding/TradingSystem/data/csv/Smarket.csv");
    int label_index = 6;

    float ratio = 0.1;
    int stop_index = data.size() * ratio;

    std::vector<std::vector<double>> data_shortened(data.begin(), data.begin() + stop_index);
    data = data_shortened;

    // IF WE JUST WANT TO USE SIMPLE DATA FROM TESTDATA.H
    data = data2;
    label_index = 1;
    // Separate labels from data before normalization

    std::vector<std::vector<double>> labels = vector1DtoColumnVector(getColumn(data, label_index));
    deleteColumn(data, label_index);

    // Normalize only the feature columns
    data = normalizeData(data);

    // Reattach labels to the data for splitting
    for (size_t i = 0; i < data.size(); i++) {
        data[i].push_back(labels[i][0]);
    }

    // Split the data into training and validation sets
    auto csv_data = splitData(data, 0.8);

    // Extract features and labels for training and validation
    std::vector<std::vector<double>> X_train = csv_data.first;
    std::vector<std::vector<double>> Y_train = vector1DtoColumnVector(getColumn(X_train, label_index));
    deleteColumn(X_train, label_index);

    std::vector<std::vector<double>> X_val = csv_data.second;
    std::vector<std::vector<double>> Y_val = vector1DtoColumnVector(getColumn(X_val, label_index));
    deleteColumn(X_val, label_index);

    // Neural Network Initialization
    int num_features = X_train[0].size();
    int num_labels = Y_train[0].size();

    GradientDescentOptimizer GD(0.001, 1000, 32, SQUARRED_ERROR);
    NeuralNetwork network1(&GD);
    network1.addInputLayer(num_features);
    // network1.addLayer(10, RELU, RANDOM);
    // network1.addLayer(6, RELU, RANDOM);
    network1.addLayer(num_labels, NONE, RANDOM);

    NeuroEvolutionOptimizer NE(0.3, 100, 100, SQUARRED_ERROR);
    NeuralNetwork network2(&NE);
    network2.addInputLayer(num_features);
    // network2.addLayer(10, RELU, RANDOM);
    // network2.addLayer(6, RELU, RANDOM);
    network2.addLayer(num_labels, NONE, RANDOM);

    // Fit models
    network1.fit(X_train, Y_train);
    network2.fit(X_train, Y_train);

    // Save training metrics
    std::vector<int> epochs(GD.num_epochs);
    std::iota(epochs.begin(), epochs.end(), 1);
    toCSV("training_loss.txt", epochs, GD.epoch_losses, GD.gradient_norms);

    // Evaluate models
    evaluateModel(network1, X_val, Y_val);
    evaluateModel(network2, X_val, Y_val);

    return 0;
}
