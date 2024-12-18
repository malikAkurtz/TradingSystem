#include <vector>
#include <iostream>
#include "Output.h"
#include "Neuron.h"
#include "NetworkLayers.h"


class NeuralNetwork {
    public:
    std::vector<std::shared_ptr<HiddenLayer>> hiddenLayers; // Use shared_ptr for polymorphism
    std::shared_ptr<InputLayer> inputLayer;                 // Input layer as a smart pointer
    std::shared_ptr<OutputLayer> outputLayer;
    int num_features;
    double loss;


    NeuralNetwork() : inputLayer(nullptr), outputLayer(nullptr){
    }


    // void fit(std::vector<std::vector<double>> featuresMatrix, std::vector<double> labels) {
    //     int num_samples = featuresMatrix.size();
    //     // output layer gradient calculation
    //     std::vector<double> a_L = this->getPredictions(featuresMatrix);
    //     double loss = calculateMSE(a_L, labels);
    //     this->loss = loss;
        
    //     std::vector<double> pd_loss_wrt_a_L = divideVector(scaleVector(subtractVectors(a_L, labels), 2), num_samples);

    //     printVector(pd_loss_wrt_a_L);
    //     std::cout << this->hiddenAndOutputLayers.size()-1 << std::endl;


    //     std::vector<double> pd_loss_wrt_z_L = matrixToVector(
    //         matrixMultiply(takeTranspose(this->hiddenAndOutputLayers.back().weightsMatrix), vectorToMatrix(pd_loss_wrt_a_L))
    //     );

    //     printVector(pd_loss_wrt_z_L);

    //     std::cout << "Here" << std::endl;
    //     std::vector<double> pd_loss_wrt_W_L = matrixToVector((vectorToMatrix(pd_loss_wrt_z_L), vectorToMatrix(this->hiddenAndOutputLayers[this->hiddenAndOutputLayers.size()-1-2].activation_outputs)));

    //     std::vector<double> prev_pd_loss_wrt_z_l = pd_loss_wrt_z_L;
    //     // hidden layers gradients calculations
    //     for (int i = (this->hiddenAndOutputLayers.size()-1); i >= 0; i--) {
    //         std::vector<double> rhs = hadamardProduct(prev_pd_loss_wrt_z_l, this->hiddenAndOutputLayers[i]
    //     .applyActivationDerivative(this->hiddenAndOutputLayers[i].pre_activation_outputs));
    //         std::vector<double> pd_loss_wrt_z_l = matrixToVector(matrixMultiply(takeTranspose(this->hiddenAndOutputLayers[i].weightsMatrix), vectorToMatrix(rhs)));
    //         if (i - 2 >= 0) {
    //             std::vector<double> pd_loss_wrt_W_l = matrixToVector(matrixMultiply(vectorToMatrix(pd_loss_wrt_z_l), 
    //             takeTranspose(vectorToMatrix(this->hiddenAndOutputLayers[i-1].activation_outputs))));
    //         } else {
    //             std::vector<double> pd_loss_wrt_W_l = matrixToVector(matrixMultiply(vectorToMatrix(pd_loss_wrt_z_l), 
    //             takeTranspose(vectorToMatrix(this->inputLayer.activation_outputs))));
    //         }
    //         std::vector<double> prev_pd_loss_wrt_z_l = pd_loss_wrt_z_l;
    //     }
        
    // }

    

    std::vector<double> getPredictions(std::vector<std::vector<double>> featuresMatrix) {

        int num_samples = featuresMatrix.size();
        std::vector<double> predictions(num_samples);

        // for every sample
        for (int i = 0; i < num_samples; i++) {
            // pass the sample into the input layer and get output
            this->inputLayer->calculateLayerOutputs(featuresMatrix[i]);
            std::vector<double> input_layer_output = this->inputLayer->getPreActivationOutputs();
            print("--------------");
            std::cout << "Input Layer Output: " << std::endl;
            printVector(input_layer_output);
            print("--------------");
            // for every hidden layer in the network
            std::vector<double> prev_layer_output = input_layer_output;
            for (int j = 0; j < this->hiddenLayers.size(); j++) {
                this->hiddenLayers[j]->calculateLayerOutputs(prev_layer_output);
                std::vector<double> this_layer_output = this->hiddenLayers[j]->getActivationOutputs();
                print("--------------");
                std::cout << "This Layer Output: " << std::endl;
                printVector(this_layer_output);
                print("--------------");
                prev_layer_output = this_layer_output;
            }

            // final pass into output layer
            this->outputLayer->calculateLayerOutputs(prev_layer_output);
            std::vector<double> output_layer_output = this->outputLayer->getActivationOutputs();
            print("--------------");
            print("Output Layer Output");
            printVector(output_layer_output);
            print("--------------");
            predictions[i] = (output_layer_output[0]); // assuming only one output neuron
        }

        return predictions;
    }

    void addHiddenLayer(std::shared_ptr<HiddenLayer> hiddenLayer) {
        this->hiddenLayers.push_back(hiddenLayer);
    }

    void addInputLayer(std::shared_ptr<InputLayer> inputLayer) {
        this->inputLayer = inputLayer;
    }

    void addOutputLayer(std::shared_ptr<OutputLayer> outputLayer) {
        this->outputLayer = outputLayer;
    }
};