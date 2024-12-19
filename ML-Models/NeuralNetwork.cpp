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
        std::vector<std::vector<std::vector<double>>> pd_J_rsp_all_layers_weights;
        int num_samples = featuresMatrix.size();
        
        for (int e = 0; e < this->num_epochs; e++) {
            double epoch_average_loss = 0;
            // for every sample
            for (int i = 0; i < num_samples; i++) {
                std::cout << "Processing sample: " << i << std::endl;
                print("----------------------------------------------------------------------------------------------------------------------------------");
                pd_J_rsp_all_layers_weights.clear();
                refreshLayers();
                // FORWARD PASS
                std::vector<double> X = featuresMatrix[i];
                print("Sample being processed");
                printVector(X);

                double y = labels[i];
                print("Sample Label");
                std::cout << y << std::endl;

                std::vector<double> A = getPredictions({X});
                print("Sample Predictions");
                printVector(A); // will just be a vector with one prediction in it since this is just for one output neuron at the moment

                // BACKWARD PASS

                // compute error + gradient at output layer, because were processing one sample at a time, m = 1, so loss = cost = squarred error
                print("Sample Squarred Error i.e Loss");
                double loss = modifiedSquarredError(A, {y}); // gradient is just A - y since the ^2 and 1/2 cancel
                std::cout << loss << std::endl;
                epoch_average_loss += loss;

                // begin back propogation starting at the output layer
                // Begin by finding partial derivative of J with respect to Z_L i.e error term at L
                std::vector<double> pd_J_rsp_Z_L = hadamardProduct(subtractVectors(A, {y}), d_ReLU(this->outputLayer->pre_activation_outputs));
                print("pd_J_rsp_Z_L");
                printVector(pd_J_rsp_Z_L);
                printVectorShape(pd_J_rsp_Z_L);

                // Compute gradients for weights
                int last_hidden_layer_index = this->hiddenLayers.size()-1;

                std::vector<double> A_hidden_with_bias = this->hiddenLayers[last_hidden_layer_index]->activation_outputs;
                A_hidden_with_bias.push_back(1);

                std::vector<std::vector<double>> pd_J_rsp_W_L = matrixMultiply(vector1Dto2D(pd_J_rsp_Z_L), 
                takeTranspose(vector1Dto2D(A_hidden_with_bias)));

                print("pd_J_rsp_W_L");
                printMatrix(pd_J_rsp_W_L);
                printMatrixShape(pd_J_rsp_W_L);
                pd_J_rsp_all_layers_weights.push_back(pd_J_rsp_W_L);
                

                // Propogate gradients to previous layers starting with last hidden layer and stopping at input layer
                std::vector<double> prev_error_term = pd_J_rsp_Z_L;

                for (int j = last_hidden_layer_index; j >= 0; j--) {
                    std::vector<double> pd_J_rsp_Z_l;
                    std::vector<std::vector<double>> pd_J_rsp_W_l;

                    std::vector<double> A_hidden_with_bias = this->hiddenLayers[last_hidden_layer_index]->activation_outputs;
                    A_hidden_with_bias.push_back(1); // Use locally, don't modify the layer's state

                    std::vector<double> DA_hidden_with_no_bias = this->hiddenLayers[j]->getDerivativeActivationOutputs();
                    
                    if (j == last_hidden_layer_index) {
                        print("prev_error_term shape");
                        printVectorShape(prev_error_term);

                        print("DA_hidden_no_bias shape");
                        printVectorShape(DA_hidden_with_no_bias);

                        print("Weight matrix shape");
                        printMatrixShape(this->outputLayer->getWeightsMatrix());

                        auto weighted_error_term = vector2Dto1D(
                            matrixMultiply(
                                takeTranspose(this->outputLayer->getWeightsMatrix()), 
                                vector1Dto2D(prev_error_term)));

                        // Remove the bias contribution (last element).
                        weighted_error_term.pop_back();

                        print("weighted_error_term");
                        printVector(weighted_error_term);

                        pd_J_rsp_Z_l = hadamardProduct(
                            weighted_error_term,
                                DA_hidden_with_no_bias);

                        prev_error_term = pd_J_rsp_Z_l;
                    } else {
                        std::cout<<"Here"<<std::endl;
                        pd_J_rsp_Z_l = hadamardProduct(vector2Dto1D(matrixMultiply(takeTranspose(this->hiddenLayers[j+1]->getWeightsMatrix()), vector1Dto2D(prev_error_term))),
                        DA_hidden_with_no_bias);
                        prev_error_term = pd_J_rsp_Z_l;
                    }
                    print("pd_J_rsp_Z_l");
                    printVector(pd_J_rsp_Z_l);
                    printVectorShape(pd_J_rsp_Z_l);


                    // calculate gradients
                    std::vector<double> A_less_1_hidden_with_bias = this->hiddenLayers[j-1]->activation_outputs;
                    A_less_1_hidden_with_bias.push_back(1);

                    pd_J_rsp_W_l = matrixMultiply(vector1Dto2D(pd_J_rsp_Z_l), 
                        takeTranspose(vector1Dto2D(A_less_1_hidden_with_bias)));

                    print("pd_J_rsp_W_l");
                    printMatrix(pd_J_rsp_W_l);
                    printMatrixShape(pd_J_rsp_W_l);
                    pd_J_rsp_all_layers_weights.push_back(pd_J_rsp_W_l);
                }


                // update weights for output layer
                this->outputLayer->updateNeuronWeights(pd_J_rsp_all_layers_weights[0], this->learning_rate);
                // update weights for every hidden layer
                for (int m = 0; m < this->hiddenLayers.size(); m++) {
                    this->hiddenLayers[m]->updateNeuronWeights(pd_J_rsp_all_layers_weights[m+1], this->learning_rate); //+1 since we already processed output layer
                }
                print("----------------------------------------------------------------------------------------------------------------------------------");
                
            }
            epoch_average_loss = epoch_average_loss / num_samples;
            std::cout << "Epoch: " << e << " Loss: " << epoch_average_loss << std::endl;
        }
        std::vector<double> best_predictions = getPredictions(featuresMatrix);
        this->model_loss = calculateMSE(best_predictions, labels);
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
                print("Hidden layer activation outputs");
                printVector(this->hiddenLayers[j]->activation_outputs);
                std::vector<double> this_layer_output = this->hiddenLayers[j]->getActivationOutputs();
                // print("--------------");
                // std::cout << "This Layer Output: " << std::endl;
                // printVector(this_layer_output);
                // print("--------------");
                prev_layer_output = this_layer_output;
            }

            // final pass into output layer
            this->outputLayer->calculateLayerOutputs(prev_layer_output);
            print("Output layer activation outputs");
            printVector(this->outputLayer->activation_outputs);
            std::vector<double> output_layer_output = this->outputLayer->getActivationOutputs();
            // print("--------------");
            // print("Output Layer Output");
            // printVector(output_layer_output);
            // print("--------------");
            predictions[i] = output_layer_output[0]; // assuming only one output neuron
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

    void refreshLayers(){
        this->outputLayer->activation_outputs.clear();
        this->outputLayer->pre_activation_outputs.clear();
        this->outputLayer->derivative_activation_outputs.clear();
        for (int i = 0; i < this->hiddenLayers.size();i++) {
            this->hiddenLayers[i]->activation_outputs.clear();
            this->hiddenLayers[i]->pre_activation_outputs.clear();
            this->hiddenLayers[i]->derivative_activation_outputs.clear();
        }
        this->inputLayer->pre_activation_outputs.clear();
    }
};