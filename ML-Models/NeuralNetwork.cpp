#include <vector>
#include <iostream>
#include <limits>
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
    double model_loss = INFINITY;


    NeuralNetwork(float learningrate, int num_epochs) : inputLayer(nullptr), outputLayer(nullptr){
        this->learning_rate = learningrate;
        this->num_layers = 0;
        this->num_epochs = num_epochs;
    }


    void fit(std::vector<std::vector<double>> featuresMatrix, std::vector<std::vector<double>> labels) {
        std::vector<std::vector<std::vector<double>>> pd_Ci_rsp_all_layers_weights;
        int num_samples = featuresMatrix.size();
        
        for (int e = 0; e < this->num_epochs; e++) {
            print("-----------------------------------------------------NEW EPOCH-----------------------------------------------------------------");
            double epoch_accumulated_loss = 0;
            // for every sample
            for (int i = 0; i < num_samples; i++) {
                pd_Ci_rsp_all_layers_weights.clear();
                // std::cout << "Processing sample: " << i << std::endl;
                // print("----------------------------------------------------------------------------------------------------------------------------------");
                // FORWARD PASS
                print("-----------------------------------------------------Forward Pass-----------------------------------------------------------------");
                std::vector<std::vector<double>> X = vector1DtoColumnVector(featuresMatrix[i]);
                print("Sample being processed");
                printMatrix(X);
                printMatrixShape(X);

                std::vector<std::vector<double>> Y = vector1DtoColumnVector(labels[i]);
                print("Sample Label");
                printMatrix(Y);
                printMatrixShape(Y);
                // this particular prediction for his 1 sample will be returned as a column vector i.e a matrix of shape (x, 1)
                // but it will be stored inside of another matrix
                // essentially its like passing in a featureMatrix with just one row (one sample) into get predictions and just getting the first 
                // and only thing that is returned by the function which would be the predictions
                std::vector<std::vector<double>> A_L = getPredictions({X})[0];
                print("Sample Predictions");
                printMatrix(A_L);
                printMatrixShape(A_L);

                // BACKWARD PASS
                print("-----------------------------------------------------Backward Pass-----------------------------------------------------------------");
                // compute error + gradient at output layer, because were processing one sample at a time, m = 1, so loss = cost = squarred error
                // essentially our loss is going to be the column vector of predictions - column vector of labels, and take the dot product
                // of the resultant which itself to get the squarred error
                double loss = modifiedSquarredError(A_L, Y); // gradient is just A - y since the ^2 and 1/2 cancel
                print("Sample Squarred Error i.e Loss i.e Cost of ith sample");
                //std::cout << loss << std::endl;
                epoch_accumulated_loss += loss;

                // begin back propogation starting at the output layer
                // gradient of loss of this particular sample with respect to loss will be a vector of matrices
                std::vector<std::vector<double>> pd_Ci_W_L;
                std::vector<std::vector<double>> error_term_L = subtractColumnVectors(A_L, Y);

                print("error term for output layer");
                printMatrix(error_term_L);
                printMatrixShape(error_term_L);

                int last_hidden_layer_index = this->hiddenLayers.size()-1;
                std::vector<std::vector<double>> A_hidden_with_bias;

                if (last_hidden_layer_index >= 0) {
                    A_hidden_with_bias = vector1DtoColumnVector(this->hiddenLayers[last_hidden_layer_index]->activation_outputs);
                } else{
                    A_hidden_with_bias = vector1DtoColumnVector(this->inputLayer->getPreActivationOutputs());
                }
                A_hidden_with_bias.push_back({1}); // adding a one to the bottom of the column vector (for bias term)

                print("vector1Dto2D(A_hidden_with_bias)");
                printMatrix(A_hidden_with_bias);
                printMatrixShape(A_hidden_with_bias);

                print("takeTranspose(A_hidden_with_bias)");
                printMatrix(takeTranspose(A_hidden_with_bias));
                printMatrixShape(takeTranspose(A_hidden_with_bias));

                pd_Ci_W_L = outerProduct(error_term_L, takeTranspose(A_hidden_with_bias)); // LOL check notes for why taking transpose

                print("pd_Ci_W_L");
                printMatrix(pd_Ci_W_L);
                printMatrixShape(pd_Ci_W_L);

                pd_Ci_rsp_all_layers_weights.push_back(pd_Ci_W_L);
                

                // Propogate gradients to previous layers starting with last hidden layer and stopping at input layer
                std::vector<std::vector<double>> prev_error_term = error_term_L;


                // for every hidden layer
                for (int j = last_hidden_layer_index; j >= 0; j--) {
                    std::vector<std::vector<double>> pd_Ci_W_l;
                    std::vector<std::vector<double>> error_term;
                    std::vector<std::vector<double>> W_l_plus_1_T;

                    if (j == last_hidden_layer_index) {
                        W_l_plus_1_T = takeTranspose(this->outputLayer->getWeightsMatrix());
                    } else {
                        W_l_plus_1_T = takeTranspose(this->hiddenLayers[j+1]->getWeightsMatrix());
                    }

                    print("W_l_plus_1_T");
                    printMatrix(W_l_plus_1_T);
                    printMatrixShape(W_l_plus_1_T);
                    
                    
                    std::vector<std::vector<double>> error_term_lhs = matrixMultiply(W_l_plus_1_T, prev_error_term);

                    error_term_lhs.pop_back(); // remove the bias contribution

                    print("error_term_lhs post bias removal");
                    printMatrix(error_term_lhs);
                    printMatrixShape(error_term_lhs);


                    std::vector<std::vector<double>> DA_hidden_with_no_bias = vector1DtoColumnVector(this->hiddenLayers[j]->getDerivativeActivationOutputs());
                    
                    print("DA_hidden_with_no_bias");
                    printMatrix(DA_hidden_with_no_bias);
                    printMatrixShape(DA_hidden_with_no_bias);

                    
                    error_term = hadamardProduct(error_term_lhs, DA_hidden_with_no_bias);

                    print("This Error Term");
                    printMatrix(error_term);
                    printMatrixShape(error_term);

                    if (j == 0) { // need to get input layer outputs
                        A_hidden_with_bias = vector1DtoColumnVector(this->inputLayer->getPreActivationOutputs());
                    } else {
                        A_hidden_with_bias = vector1DtoColumnVector(this->hiddenLayers[j-1]->activation_outputs);
                        
                    }
                    A_hidden_with_bias.push_back({1}); 

                    print("A_hidden_with_bias");
                    printMatrix(A_hidden_with_bias);
                    printMatrixShape(A_hidden_with_bias);
                    
                    pd_Ci_W_l = matrixMultiply(error_term, takeTranspose(A_hidden_with_bias));
                    
                    print("pd_Ci_W_l");
                    printMatrix(pd_Ci_W_l);
                    printMatrixShape(pd_Ci_W_l);

                    pd_Ci_rsp_all_layers_weights.push_back(pd_Ci_W_l);
                    prev_error_term = error_term;
                }


                // update weights for output layer
                printMatrix(pd_Ci_rsp_all_layers_weights[0]);
                std::cout << "Here" << std::endl;
                this->outputLayer->updateNeuronWeights(pd_Ci_rsp_all_layers_weights[0], this->learning_rate);
                std::cout << "Here2" << std::endl;
                // update weights for every hidden layer
                for (int m = 0; m < this->hiddenLayers.size(); m++) {
                    int gradient_index = pd_Ci_rsp_all_layers_weights.size() - 1 - m;
                    this->hiddenLayers[m]->updateNeuronWeights(pd_Ci_rsp_all_layers_weights[gradient_index], this->learning_rate); //+1 since we already processed output layer
                }

                print("New Hidden Layer Parameters Starting from First Hidden Layer After Processing This Sample");
                for (int i = 0; i < this->hiddenLayers.size();i++) {
                    printMatrix(this->hiddenLayers[i]->getWeightsMatrix());
                }
                print("New Output Layer Parameters After Processing This Sample");
                printMatrix(this->outputLayer->getWeightsMatrix());
                print("----------------------------------------------------------------------------------------------------------------------------------");
                
            }
            double epoch_MSE = epoch_accumulated_loss / num_samples; // mean squarred error for this epoch
            std::cout << "Epoch: " << e << " MSE: " << epoch_MSE << std::endl;
        }
        std::vector<std::vector<std::vector<double>>> best_predictions = getPredictions(featuresMatrix);
        double accumulated_final_model_loss = 0;
        for (int i = 0; i < best_predictions.size(); i++) {
            accumulated_final_model_loss += modifiedSquarredError(best_predictions[i], vector1DtoColumnVector(labels[i]));
        }
        this->model_loss = accumulated_final_model_loss / labels.size(); // mean squarred error
    }

    
    std::vector<std::vector<std::vector<double>>> getPredictions(std::vector<std::vector<double>> featuresMatrix) {
        int num_samples = featuresMatrix.size();
        std::vector<std::vector<std::vector<double>>> predictions;

        // for every sample
        for (int i = 0; i < num_samples; i++) {
            // pass the sample into the input layer and get output
            this->inputLayer->calculateLayerOutputs(featuresMatrix[i]);
            std::vector<std::vector<double>> input_layer_output = vector1DtoColumnVector(this->inputLayer->getPreActivationOutputs());
            print("--------------");
            print("Input Layer Output");
            printMatrix(input_layer_output);
            printMatrixShape(input_layer_output);
            print("--------------");
            // for every hidden layer in the network
            std::vector<std::vector<double>> prev_layer_output = input_layer_output;
            for (int j = 0; j < this->hiddenLayers.size(); j++) {
                this->hiddenLayers[j]->calculateLayerOutputs(columnVectortoVector1D(prev_layer_output));
                // print("Hidden layer activation outputs");
                // printVector(this->hiddenLayers[j]->activation_outputs);
                std::vector<std::vector<double>> this_layer_output = vector1DtoColumnVector(this->hiddenLayers[j]->getActivationOutputs());
                print("--------------");
                print("This Layer Output");
                printMatrix(this_layer_output);
                printMatrixShape(this_layer_output);
                print("--------------");
                prev_layer_output = this_layer_output;
            }

            // final pass into output layer
            this->outputLayer->calculateLayerOutputs(columnVectortoVector1D(prev_layer_output));
            // print("Output layer activation outputs");
            // printVector(this->outputLayer->activation_outputs);
            std::vector<std::vector<double>> output_layer_output = vector1DtoColumnVector(this->outputLayer->getActivationOutputs());
            print("--------------");
            print("Output Layer Output");
            printMatrix(output_layer_output);
            printMatrixShape(output_layer_output);
            print("--------------");
            predictions.push_back(output_layer_output);
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