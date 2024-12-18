#include <vector>
#include <iostream>
#include "Output.h"
#include "Neuron.h"
#include "NetworkLayers.h"


class NeuralNetwork {
    public:
    float learning_rate;
    int num_epochs;
    int num_layers ;
    std::vector<std::shared_ptr<HiddenLayer>> hiddenLayers; // Use shared_ptr for polymorphism
    std::shared_ptr<InputLayer> inputLayer;                 // Input layer as a smart pointer
    std::shared_ptr<OutputLayer> outputLayer;
    int num_features;
    double model_loss;


    NeuralNetwork(float learningrate, int num_epochs) : inputLayer(nullptr), outputLayer(nullptr){
        this->learning_rate = learningrate;
        this->num_layers = 0;
        this->num_epochs = num_epochs;
    }


    void fit(std::vector<std::vector<double>> featuresMatrix, std::vector<double> labels) {
        int num_samples = featuresMatrix.size();
        
        for (int e = 0; e < this->num_epochs; e++) {
            double epoch_average_loss = 0;
            // for every sample
            for (int i = 0; i < num_samples; i++) {
                // forward propogate it
                std::vector<double> sample = featuresMatrix[i];
                print("Sample being processed");
                printVector(sample);
                print("Sample Prediction");
                std::vector<double> prediction = getPredictions({sample});
                printVector(prediction);
                double label = labels[i];
                print("Sample Label");
                std::cout << label << std::endl;
                print("Loss");
                std::cout << calculateMSE_Simple({prediction}, {label}) << std::endl;
                epoch_average_loss += calculateMSE_Simple({prediction}, {label});
                int last_layer_index = num_layers - 3;

                // calculate partial derivative of loss with respect to output layer weights
                std::vector<double> dL_rsp_W_output;
                std::vector<double> dL_resp_z_output = createVector(calculateMSE_Simple({prediction}, {label}), this->hiddenLayers[last_layer_index]->neurons.size());
                print("dL_resp_z_output");
                printVector(dL_resp_z_output);
                std::vector<double> dz_output_rsp_W_output = this->hiddenLayers[last_layer_index]->getActivationOutputs();
                print("dz_output_rsp_W_output");
                printVector(dz_output_rsp_W_output);
                dL_rsp_W_output = hadamardProduct(dz_output_rsp_W_output, dL_resp_z_output);
                print("dL_rsp_W_output");
                printVector(dL_rsp_W_output);
                

                // calculate partial derivative of loss with respect to previous layer weights
                for (int j = last_layer_index; j >= 0; j-- ) {
                    // print("this->hiddenLayers[last_layer_index]->neurons.size()");
                    // std::cout << this->hiddenLayers[i]->neurons.size() << std::endl;
                    // print("std::vector<double>(sample.size()");
                    // std::cout << sample.size() << std::endl;
                    std::vector<std::vector<double>> dL_rsp_W_hidden(this->hiddenLayers[j]->neurons.size(), std::vector<double>(sample.size()));
                    std::vector<double> dL_rsp_z_hidden = matrixToVector(
                        matrixMultiply(takeTranspose(this->outputLayer->getWeightsMatrix()), takeTranspose(vectorToMatrix(dL_resp_z_output))));
                    print("dL_rsp_z_hidden");
                    printVector(dL_rsp_z_hidden);
                    std::vector<double> dz_hidden_rsp_W_hidden = sample;
                    print("dz_hidden_rsp_W_hidden");
                    printVector(dz_hidden_rsp_W_hidden);
                    dL_rsp_W_hidden = matrixMultiply(vectorToMatrix(dL_rsp_z_hidden), takeTranspose(vectorToMatrix(dz_hidden_rsp_W_hidden))); // tranpose ????
                    print("dL_rsp_W_hidden");
                    printMatrix(dL_rsp_W_hidden);


                    // update hidden layer weights
                    for (int k = 0; k < this->hiddenLayers[last_layer_index]->weightsMatrix.size(); ++k) {
                        for (int l = 0; l < this->hiddenLayers[last_layer_index]->weightsMatrix[0].size(); ++l) {
                            this->hiddenLayers[last_layer_index]->weightsMatrix[k][l] -= (learning_rate * dL_rsp_W_hidden[k][l]);
                        }
                    }
                }
            }
            epoch_average_loss = epoch_average_loss / num_samples;
            std::cout << "Epoch: " << e << " Loss: " << epoch_average_loss << std::endl;
        }
        std::vector<double> best_predictions = getPredictions(featuresMatrix);
        this->model_loss = calculateMSE_Simple(best_predictions, labels);
    }

    

    std::vector<double> getPredictions(std::vector<std::vector<double>> featuresMatrix) {
        int num_samples = featuresMatrix.size();
        std::vector<double> predictions(num_samples);

        // for every sample
        for (int i = 0; i < num_samples; i++) {
            // pass the sample into the input layer and get output
            this->inputLayer->calculateLayerOutputs(featuresMatrix[i]);
            std::vector<double> input_layer_output = this->inputLayer->getPreActivationOutputs();
            // print("--------------");
            // std::cout << "Input Layer Output: " << std::endl;
            // printVector(input_layer_output);
            // print("--------------");
            // for every hidden layer in the network
            std::vector<double> prev_layer_output = input_layer_output;
            for (int j = 0; j < this->hiddenLayers.size(); j++) {
                this->hiddenLayers[j]->calculateLayerOutputs(prev_layer_output);
                std::vector<double> this_layer_output = this->hiddenLayers[j]->getActivationOutputs();
                // print("--------------");
                // std::cout << "This Layer Output: " << std::endl;
                // printVector(this_layer_output);
                // print("--------------");
                prev_layer_output = this_layer_output;
            }

            // final pass into output layer
            this->outputLayer->calculateLayerOutputs(prev_layer_output);
            std::vector<double> output_layer_output = this->outputLayer->getActivationOutputs();
            // print("--------------");
            // print("Output Layer Output");
            // printVector(output_layer_output);
            // print("--------------");
            predictions[i] = (output_layer_output[0]); // assuming only one output neuron
        }

        return predictions;
    }

    void addHiddenLayer(std::shared_ptr<HiddenLayer> hiddenLayer) {
        this->hiddenLayers.push_back(hiddenLayer);
        this->num_layers += 1;
    }

    void addInputLayer(std::shared_ptr<InputLayer> inputLayer) {
        this->inputLayer = inputLayer;
        this->num_layers += 1;
    }

    void addOutputLayer(std::shared_ptr<OutputLayer> outputLayer) {
        this->outputLayer = outputLayer;
        this->num_layers += 1;
    }
};