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


    void fit(std::vector<std::vector<double>> featuresMatrix, std::vector<std::vector<double>>  labels) {
        //convert from 1d vectors to 2d column vectors iniside the function
        std::vector<std::vector<std::vector<double>>> pd_Ci_rsp_all_layers_weights; // matrix of matrices representing gradients
        int num_samples = featuresMatrix.size();
        
        for (int e = 0; e < this->num_epochs; e++) {
            printDebug("-----------------------------------------------------NEW EPOCH-----------------------------------------------------------------");
            double epoch_accumulated_loss = 0;
            // for every sample
            for (int i = 0; i < num_samples; i++) {
                pd_Ci_rsp_all_layers_weights.clear();
                // print("----------------------------------------------------------------------------------------------------------------------------------");
                // FORWARD PASS
                printDebug("-----------------------------------------------------Forward Pass-----------------------------------------------------------------");
                std::vector<std::vector<double>> X = vector1DtoColumnVector(featuresMatrix[i]); // featuresMatrix[i] is a 1d vector of features
                // X will be used to transform the features for this particular sample into a column vector to be used in computation
                printDebug("Sample being processed");
                printMatrixDebug(X);
                printMatrixShapeDebug(X);

                std::vector<std::vector<double>> Y = vector1DtoColumnVector(labels[i]); // labels[i] is a 1d vector of labels
                // Y will be used to transform the labels for this particular sample into a column vector to be used in computation
                printDebug("Sample Label");
                printMatrixDebug(Y);
                printMatrixShapeDebug(Y);
                // this particular prediction for his 1 sample will be returned as a column vector i.e a matrix of shape (x, 1)
                // but it will be stored inside of another matrix
                // essentially its like passing in a featureMatrix with just one row (one sample) into get predictions and just getting the first 
                // and only thing that is returned by the function which would be the predictions
                // getPredictions takes in a data set matrix, so pass in the row as a matrix, essentially a dataset with 1 row i.e 1 sample
                // getPredictions returns a vector of Column vectors representing predictions, so get the first one and only one
                // so A_L will automatically be a column vector
                std::vector<std::vector<double>> A_L = getPredictions({featuresMatrix[i]})[0];
                printDebug("Sample Predictions");
                printMatrixDebug(A_L);
                printMatrixShapeDebug(A_L);

                // BACKWARD PASS
                printDebug("-----------------------------------------------------Backward Pass-----------------------------------------------------------------");
                // compute error + gradient at output layer, because were processing one sample at a time, m = 1, so loss = cost = squarred error
                // essentially our loss is going to be the column vector of predictions - column vector of labels, and take the dot product
                // of the resultant which itself to get the squarred error
                double loss = modifiedSquarredError(A_L, Y); // gradient is just A - y since the ^2 and 1/2 cancel
                printDebug("Sample Squarred Error i.e Loss i.e Cost of ith sample");
                printDebug(loss);
                epoch_accumulated_loss += loss;

                // begin back propogation starting at the output layer
                // gradient of loss of this particular sample with respect to loss will be a vector of matrices
                std::vector<std::vector<double>> pd_Ci_W_L;
                std::vector<std::vector<double>> error_term_L = subtractColumnVectors(A_L, Y);

                printDebug("error term for output layer");
                printMatrixDebug(error_term_L);
                printMatrixShapeDebug(error_term_L);

                int last_hidden_layer_index = this->hiddenLayers.size()-1;
                std::vector<std::vector<double>> A_hidden_with_bias;

                if (last_hidden_layer_index >= 0) {
                    A_hidden_with_bias = this->hiddenLayers[last_hidden_layer_index]->getActivationOutputs();
                } else{
                    A_hidden_with_bias = this->inputLayer->getPreActivationOutputs();
                }
                A_hidden_with_bias.push_back({1}); // adding a one to the bottom of the column vector (for bias term)

                printDebug("vector1Dto2D(A_hidden_with_bias)");
                printMatrixDebug(A_hidden_with_bias);
                printMatrixShapeDebug(A_hidden_with_bias);

                printDebug("takeTranspose(A_hidden_with_bias)");
                printMatrixDebug(takeTranspose(A_hidden_with_bias));
                printMatrixShapeDebug(takeTranspose(A_hidden_with_bias));

                pd_Ci_W_L = outerProduct(error_term_L, takeTranspose(A_hidden_with_bias)); // LOL check notes for why taking transpose

                printDebug("pd_Ci_W_L");
                printMatrixDebug(pd_Ci_W_L);
                printMatrixShapeDebug(pd_Ci_W_L);

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

                    printDebug("W_l_plus_1_T");
                    printMatrixDebug(W_l_plus_1_T);
                    printMatrixShapeDebug(W_l_plus_1_T);
                    
                    
                    std::vector<std::vector<double>> error_term_lhs = matrixMultiply(W_l_plus_1_T, prev_error_term);

                    error_term_lhs.pop_back(); // remove the bias contribution

                    printDebug("error_term_lhs post bias removal");
                    printMatrixDebug(error_term_lhs);
                    printMatrixShapeDebug(error_term_lhs);


                    std::vector<std::vector<double>> DA_hidden_with_no_bias = this->hiddenLayers[j]->getDerivativeActivationOutputs();
                    
                    printDebug("DA_hidden_with_no_bias");
                    printMatrixDebug(DA_hidden_with_no_bias);
                    printMatrixShapeDebug(DA_hidden_with_no_bias);

                    
                    error_term = hadamardProduct(error_term_lhs, DA_hidden_with_no_bias);

                    printDebug("This Error Term");
                    printMatrixDebug(error_term);
                    printMatrixShapeDebug(error_term);

                    if (j == 0) { // need to get input layer outputs
                        A_hidden_with_bias = this->inputLayer->getPreActivationOutputs();
                    } else {
                        A_hidden_with_bias = this->hiddenLayers[j-1]->getActivationOutputs();
                        
                    }
                    A_hidden_with_bias.push_back({1}); 

                    printDebug("A_hidden_with_bias");
                    printMatrixDebug(A_hidden_with_bias);
                    printMatrixShapeDebug(A_hidden_with_bias);
                    std::vector<std::vector<double>> A_hidden_with_bias_T = takeTranspose(A_hidden_with_bias);
                    pd_Ci_W_l = matrixMultiply(error_term, A_hidden_with_bias_T);
                    
                    printDebug("pd_Ci_W_l");
                    printMatrixDebug(pd_Ci_W_l);
                    printMatrixShapeDebug(pd_Ci_W_l);

                    pd_Ci_rsp_all_layers_weights.push_back(pd_Ci_W_l);
                    prev_error_term = error_term;
                }


                // update weights for output layer
                printMatrixDebug(pd_Ci_rsp_all_layers_weights[0]);
                this->outputLayer->updateNeuronWeights(pd_Ci_rsp_all_layers_weights[0], this->learning_rate);
                // update weights for every hidden layer
                for (int m = 0; m < this->hiddenLayers.size(); m++) {
                    int gradient_index = pd_Ci_rsp_all_layers_weights.size() - 1 - m;
                    this->hiddenLayers[m]->updateNeuronWeights(pd_Ci_rsp_all_layers_weights[gradient_index], this->learning_rate); //+1 since we already processed output layer
                }

                printDebug("New Hidden Layer Parameters Starting from First Hidden Layer After Processing This Sample");
                for (int i = 0; i < this->hiddenLayers.size();i++) {
                    printMatrixDebug(this->hiddenLayers[i]->getWeightsMatrix());
                }
                printDebug("New Output Layer Parameters After Processing This Sample");
                printMatrixDebug(this->outputLayer->getWeightsMatrix());
                printDebug("-------------------------------------------------END EPOCH-------------------------------------------------------------");
                
            }
            double epoch_MSE = epoch_accumulated_loss / num_samples; // mean squarred error for this epoch
            std::cout << "Epoch: " << e << " MSE: " << epoch_MSE << std::endl;
        }
        // now getting predictions of the entire feature matrix, i.e all samples
        // best_predictions will then consist of a vector of column vectors
        std::vector<std::vector<std::vector<double>>> best_predictions = getPredictions(featuresMatrix);
        double accumulated_final_model_loss = 0;
        std::vector<std::vector<std::vector<double>>> labels_as_col_vectors;
        for (int i = 0; i < labels.size(); i++) {
            labels_as_col_vectors.push_back(vector1DtoColumnVector(labels[i]));
        }
        for (int i = 0; i < best_predictions.size(); i++) {
            accumulated_final_model_loss += modifiedSquarredError(best_predictions[i], labels_as_col_vectors[i]);
        }
        this->model_loss = accumulated_final_model_loss / labels.size(); // mean squarred error
    }

    // currently this returns a matrix of column vectors representing the outputs for each forward pass of each sample
    // takes in a VECTOR of COLUMN VECTORS
    std::vector<std::vector<std::vector<double>>> getPredictions(std::vector<std::vector<double>> featuresMatrix) {
        printDebug("Features Matrix looks like");
        printMatrixDebug(featuresMatrix);
        int num_samples = featuresMatrix.size();
        printDebug("number of samples");
        printDebug(num_samples);
        std::vector<std::vector<std::vector<double>>> predictions;

        // for every sample
        for (int i = 0; i < num_samples; i++) {
            // pass the sample into the input layer and get output
            this->inputLayer->calculateLayerOutputs(vector1DtoColumnVector(featuresMatrix[i]));
            std::vector<std::vector<double>> input_layer_output = this->inputLayer->getPreActivationOutputs();
            printDebug("------------------------------------------------Getting Predictions--------------------------------------------------------------");
            printDebug("Input Layer Output");
            printMatrixDebug(input_layer_output);
            printMatrixShapeDebug(input_layer_output);
            // for every hidden layer in the network
            std::vector<std::vector<double>> prev_layer_output = input_layer_output;
            for (int j = 0; j < this->hiddenLayers.size(); j++) {
                this->hiddenLayers[j]->calculateLayerOutputs(prev_layer_output);
                // print("Hidden layer activation outputs");
                // printVector(this->hiddenLayers[j]->activation_outputs);
                std::vector<std::vector<double>> this_layer_output = this->hiddenLayers[j]->getActivationOutputs();
                printDebug("This Layer Output");
                printMatrixDebug(this_layer_output);
                printMatrixShapeDebug(this_layer_output);
                prev_layer_output = this_layer_output;
            }

            // final pass into output layer
            this->outputLayer->calculateLayerOutputs(prev_layer_output);
            // print("Output layer activation outputs");
            // printVector(this->outputLayer->activation_outputs);
            std::vector<std::vector<double>> output_layer_output = this->outputLayer->getActivationOutputs();
            printDebug("Output Layer Output");
            printMatrixDebug(output_layer_output);
            printMatrixShapeDebug(output_layer_output);
            predictions.push_back(output_layer_output);
            printDebug("------------------------------------------------Got Prediction--------------------------------------------------------------");
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