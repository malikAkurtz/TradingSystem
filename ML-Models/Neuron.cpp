#include <vector>
#include "LinearAlgebra.h"
#include "GenFunctions.h"
#include "Output.h"
#include <iostream>

enum ActivationFunction {
    RELU,
    SIGMOID,
    NONE
};

class Neuron {
    public:
    int num_parameters;
    std::vector<double> parameters;


    Neuron(int num_params) {
        this->num_parameters = num_params;
        for (int i = 0; i < num_params; i++) {
            this->parameters.push_back(2);
        }
    }
 
    std::vector<double> getParameters(){
        return this->parameters;
    }

};


class HiddenLayer {
    public:
    std::vector<Neuron> neurons;
    ActivationFunction AFtype = RELU;
    std::vector<std::vector<double>> weightsMatrix;

    HiddenLayer(int num_neurons, int num_neurons_in_prev_layer, ActivationFunction type) {
        this->AFtype = type;
        for (int i = 0; i < num_neurons; i++) {
            Neuron defaultNeuron(num_neurons_in_prev_layer + 1); // including bias
            addNeuron(defaultNeuron);
        }
    }

    
    std::vector<double> getLayerOutput(std::vector<double> input_vector) {
        std::vector<double> padded_input = input_vector;
        padded_input.push_back(1); // essentially a constant to multiply with the bias when taking the dot product
        print("------------------------------------------------------------------");
        print("Padded Input Vector: ");
        printVector(padded_input);
        print("Weights Matrix: ");
        printMatrix(this->weightsMatrix);
        std::vector<double> pre_activation_output = matrixToVector(matrixMultiply(this->weightsMatrix, vectorToMatrix(padded_input)));
        print("Matrix Multiply Resulting Layer Output");
        printVector(pre_activation_output);
        print("------------------------------------------------------------------");
        return applyActivation(pre_activation_output);
    }

    std::vector<double> applyActivation(std::vector<double> pre_activation_output) {
        std::vector<double> layer_output(this->neurons.size());

        if (this->AFtype == RELU) {
            for (int i = 0; i < layer_output.size(); i++) {
                layer_output[i] = ReLU(pre_activation_output[i]);
            }
            return layer_output;
        } else if (this->AFtype == SIGMOID) {
            for (int i = 0; i < layer_output.size(); i++) {
                layer_output[i] = sigmoid(pre_activation_output[i]);
            }
            return layer_output;
        } else {
            return pre_activation_output;
        }

    }

    void addNeuron(Neuron neuron) {
        this->neurons.push_back(neuron);
        this->weightsMatrix.push_back(neuron.getParameters());
    }
};

class InputLayer {
    public:
    std::vector<Neuron> neurons;
    int num_neurons;
    std::vector<std::vector<double>> weightsMatrix;

    InputLayer() {
        this->num_neurons = 0;
    }

    InputLayer(int num_neurons) {
        for (int i = 0; i < num_neurons; i++) {
            Neuron defaultNeuron(1);
            addNeuron(defaultNeuron);
        }
    }

    std::vector<double> getLayerOutput(std::vector<double> input_vector) {
        return input_vector;
    }

    void addNeuron(Neuron neuron) {
        this->neurons.push_back(neuron);
        this->weightsMatrix.push_back(neuron.getParameters());
    }

};

class OutputLayer {
    public:
    std::vector<Neuron> neurons;
    int num_neurons;
    std::vector<std::vector<double>> weightsMatrix;

    OutputLayer() {
        this->num_neurons = 0;
    }

    OutputLayer(int num_neurons, int num_params_per_neuron) {
        this->num_neurons = num_neurons;
        for (int i = 0; i < num_neurons; i++) {
            Neuron defaultNeuron(num_params_per_neuron);
            addNeuron(defaultNeuron);
        }
    }

    
    std::vector<double> getLayerOutput(std::vector<double> input_vector) {
        std::vector<double> output = matrixToVector(matrixMultiply(weightsMatrix, vectorToMatrix(input_vector)));
        return output;
    }


    void addNeuron(Neuron neuron) {
        this->neurons.push_back(neuron);
        this->weightsMatrix.push_back(neuron.getParameters());
    }
};

class NeuralNetwork {
    public:
    std::vector<HiddenLayer> hiddenLayers;
    InputLayer inputLayer;
    OutputLayer outputLayer;
    int num_features;


    NeuralNetwork() {
    }


    void fit(std::vector<std::vector<double>> featuresMatrix, std::vector<double> labels) {

    }

    std::vector<double> getPredictions(std::vector<std::vector<double>> featuresMatrix) {

        int num_samples = featuresMatrix.size();
        std::vector<double> predictions(num_samples);

        // for every sample
        for (int i = 0; i < num_samples; i++) {
            // pass the sample into the input layer and get output
            InputLayer& inputLayer = this->inputLayer;
            std::vector<double> input_layer_output = inputLayer.getLayerOutput(featuresMatrix[i]);
            // std::cout << "Input Layer Output: " << std::endl;
            // printVector(input_layer_output);

            // for every hidden layer in the network
            std::vector<double> prev_layer_output = input_layer_output;
            for (int j = 0; j < this->hiddenLayers.size(); j++) {
                HiddenLayer& currentLayer = this->hiddenLayers[j];
                std::vector<double> this_layer_output = currentLayer.getLayerOutput(prev_layer_output);
                // std::cout << "This Layer Output: " << std::endl;
                // printVector(this_layer_output);
                prev_layer_output = this_layer_output;
            }

            // final pass into output layer
            OutputLayer& outputLayer = this->outputLayer;
            std::vector<double> output_layer_output = outputLayer.getLayerOutput(prev_layer_output);
            // std::cout << "Output Layer Output: " << std::endl;
            // printVector(output_layer_output);
        }

        return predictions;
    }

    void addHiddenLayer(HiddenLayer hiddenLayer) {
        this->hiddenLayers.push_back(hiddenLayer);
    }

    void addInputLayer(InputLayer inputLayer) {
        this->inputLayer = inputLayer;
    }

    void addOutputLayer(OutputLayer outputLayer) {
        this->outputLayer = outputLayer;
    }
};