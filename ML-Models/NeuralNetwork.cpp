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
        std::vector<std::vector<std::vector<double>>> pd_Ci_rsp_all_layers_weights;
        int num_samples = featuresMatrix.size();
        
        for (int e = 0; e < this->num_epochs; e++) {
            // print("-----------------------------------------------------NEW EPOCH-----------------------------------------------------------------");
            double epoch_average_loss = 0;
            // for every sample
            for (int i = 0; i < num_samples; i++) {
                pd_Ci_rsp_all_layers_weights.clear();
                // std::cout << "Processing sample: " << i << std::endl;
                // print("----------------------------------------------------------------------------------------------------------------------------------");
                // FORWARD PASS
                // print("-----------------------------------------------------Forward Pass-----------------------------------------------------------------");
                std::vector<double> X = featuresMatrix[i];
                // print("Sample being processed");
                // printVector(X);

                double y = labels[i];
                // print("Sample Label");
                // std::cout << y << std::endl;

                std::vector<double> A_L = getPredictions({X});
                // print("Sample Predictions");
                // printVector(A_L); // will just be a vector with one prediction in it since this is just for one output neuron at the moment
                // printVectorShape(A_L);

                // BACKWARD PASS
                // print("-----------------------------------------------------Backward Pass-----------------------------------------------------------------");
                // compute error + gradient at output layer, because were processing one sample at a time, m = 1, so loss = cost = squarred error
                double loss = modifiedSquarredError(A_L, {y}); // gradient is just A - y since the ^2 and 1/2 cancel
                // print("Sample Squarred Error i.e Loss i.e Cost of ith sample");
                // std::cout << loss << std::endl;
                epoch_average_loss += loss;

                // begin back propogation starting at the output layer
                std::vector<std::vector<double>> pd_Ci_W_L;
                std::vector<double> error_term_L = subtractVectors(A_L, createVector(y, 1));
                // print("error term for output layer");
                // printVector(error_term_L);
                // printVectorShape(error_term_L);

                int last_hidden_layer_index = this->hiddenLayers.size()-1;
                std::vector<double> A_hidden_with_bias = this->hiddenLayers[last_hidden_layer_index]->activation_outputs;
                A_hidden_with_bias.push_back(1);

                // print("vector1Dto2D(A_hidden_with_bias)");
                // printMatrix(vector1Dto2D(A_hidden_with_bias));
                // printMatrixShape(vector1Dto2D(A_hidden_with_bias));

                // print("vector1Dto2D(error_term_L)");
                // printMatrix(vector1Dto2D(error_term_L));
                // printMatrixShape(vector1Dto2D(error_term_L));
                pd_Ci_W_L = takeTranspose(matrixMultiply(vector1Dto2D(A_hidden_with_bias), vector1Dto2D(error_term_L))); // LOL check notes for why taking transpose

                // print("pd_Ci_W_L");
                // printMatrix(pd_Ci_W_L);
                // printMatrixShape(pd_Ci_W_L);

                pd_Ci_rsp_all_layers_weights.push_back(pd_Ci_W_L);
                

                // Propogate gradients to previous layers starting with last hidden layer and stopping at input layer
                std::vector<double> prev_error_term = error_term_L;

                // print("Last Hidden Layer Index");
                // std::cout << last_hidden_layer_index << std::endl;
                // for every hidden layer
                for (int j = last_hidden_layer_index; j >= 0; j--) {
                    std::vector<std::vector<double>> pd_Ci_W_l;
                    std::vector<double> error_term;
                    std::vector<std::vector<double>> W_l_plus_1_T;

                    if (j == last_hidden_layer_index) {
                        W_l_plus_1_T = takeTranspose(this->outputLayer->getWeightsMatrix());
                    } else {
                        W_l_plus_1_T = takeTranspose(this->hiddenLayers[j+1]->getWeightsMatrix());
                    }

                    // print("W_l_plus_1_T");
                    // printMatrix(W_l_plus_1_T);
                    // printMatrixShape(W_l_plus_1_T);
                    
                    
                    std::vector<double> error_term_lhs = vector2Dto1D(matrixMultiply(W_l_plus_1_T, vector1Dto2D(prev_error_term)));
                    error_term_lhs.pop_back(); // remove the bias contribution
                    // print("error_term_lhs post bias removal");
                    // printVector(error_term_lhs);
                    // printVectorShape(error_term_lhs);


                    std::vector<double> DA_hidden_with_no_bias = this->hiddenLayers[j]->getDerivativeActivationOutputs();
                    
                    // print("DA_hidden_with_no_bias");
                    // printVector(DA_hidden_with_no_bias);
                    // printVectorShape(DA_hidden_with_no_bias);

                    
                    error_term = hadamardProduct(error_term_lhs, DA_hidden_with_no_bias);
                    // print("This Error Term");
                    // printVector(error_term);
                    // printVectorShape(error_term);

                    if (j == 0) { // need to get input layer outputs
                        A_hidden_with_bias = this->inputLayer->getPreActivationOutputs();
                        A_hidden_with_bias.push_back(1); 
                    } else {
                        A_hidden_with_bias = this->hiddenLayers[j-1]->activation_outputs;
                        A_hidden_with_bias.push_back(1); 
                    }

                    // print("A_hidden_with_bias");
                    // printVector(A_hidden_with_bias);
                    // printVectorShape(A_hidden_with_bias);
                    
                    pd_Ci_W_l = matrixMultiply(vector1Dto2D(error_term), {A_hidden_with_bias});
                    
                    // print("pd_Ci_W_l");
                    // printMatrix(pd_Ci_W_l);
                    // printMatrixShape(pd_Ci_W_l);

                    pd_Ci_rsp_all_layers_weights.push_back(pd_Ci_W_l);
                }


                // update weights for output layer
                this->outputLayer->updateNeuronWeights(pd_Ci_rsp_all_layers_weights[0], this->learning_rate);
                // update weights for every hidden layer
                for (int m = 0; m < this->hiddenLayers.size(); m++) {
                    int gradient_index = pd_Ci_rsp_all_layers_weights.size() - 1 - m;
                    this->hiddenLayers[m]->updateNeuronWeights(pd_Ci_rsp_all_layers_weights[gradient_index], this->learning_rate); //+1 since we already processed output layer
                }

                // print("New Hidden Layer Parameters Starting from First Hidden Layer After Processing This Sample");
                // for (int i = 0; i < this->hiddenLayers.size();i++) {
                //     printMatrix(this->hiddenLayers[0]->getWeightsMatrix());
                // }
                // print("New Output Layer Parameters After Processing This Sample");
                // printMatrix(this->outputLayer->getWeightsMatrix());
                // print("----------------------------------------------------------------------------------------------------------------------------------");
                
            }
            epoch_average_loss = epoch_average_loss / num_samples;
            std::cout << "Epoch: " << e << " Loss: " << epoch_average_loss << std::endl;
        }
        std::vector<double> best_predictions = getPredictions(featuresMatrix);
        this->model_loss = modifiedSquarredError(best_predictions, labels);
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
                // print("Hidden layer activation outputs");
                // printVector(this->hiddenLayers[j]->activation_outputs);
                std::vector<double> this_layer_output = this->hiddenLayers[j]->getActivationOutputs();
                // print("--------------");
                // std::cout << "This Layer Output: " << std::endl;
                // printVector(this_layer_output);
                // print("--------------");
                prev_layer_output = this_layer_output;
            }

            // final pass into output layer
            this->outputLayer->calculateLayerOutputs(prev_layer_output);
            // print("Output layer activation outputs");
            // printVector(this->outputLayer->activation_outputs);
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