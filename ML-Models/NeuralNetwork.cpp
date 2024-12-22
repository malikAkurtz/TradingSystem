#include <vector>
#include <iostream>
#include <limits>
#include "Output.h"
#include "Neuron.h"
#include "NetworkLayers.h"
#include "GenFunctions.h"


class NeuralNetwork {
    public:
    float LR;
    int num_epochs;
    int num_hidden_layers ;
    std::shared_ptr<InputLayer> inputLayer;
    std::vector<std::shared_ptr<Layer>> layers;
    int num_features;
    double model_loss = INFINITY;
    std::vector<double> epoch_losses;
    std::vector<double> epoch_gradient_norms;
    LossFunction selectedLoss;

    NeuralNetwork(float learningrate, int num_epochs, LossFunction lossFunction) : inputLayer(nullptr){
        this->LR = learningrate;
        this->num_hidden_layers = 0;
        this->num_epochs = num_epochs;
        this->selectedLoss = lossFunction;
    }

    void fit(std::vector<std::vector<double>> featuresMatrix, std::vector<std::vector<double>>  labels) {
        // ΔCᵢ, vector of matrices representing gradients with respect to each of the layers weights
        std::vector<std::vector<std::vector<double>>> gradient_Ci; 
        int num_samples = featuresMatrix.size();
        
        // for every epoch
        for (int e = 0; e < this->num_epochs; e++) {
            printDebug("--------------------NEW EPOCH--------------------");
            double epoch_accumulated_loss = 0;
            double epoch_accumulated_gradient = 0;
            // for every sample in the data set
            for (int i = 0; i < num_samples; i++) {
                // we are going to find the gradient of the loss of this particular sample with each layers weights
                gradient_Ci.clear();

                printDebug("-------------------Forward Pass----------------");
                // Store the features for this particular sample as a column vector
                std::vector<std::vector<double>> X = vector1DtoColumnVector(featuresMatrix[i]);
                printDebug("Sample being processed");
                printMatrixDebug(X);

                // Store the labels for this particular sample as a column vector
                std::vector<std::vector<double>> Y = vector1DtoColumnVector(labels[i]); 
                printDebug("Corresponding Label");
                printMatrixDebug(Y);
                /* 
                The output of the last hidden layer for this sample, Aᴸ, will be the result of indexing the first and only output of the getPredictions method, which takes in a matrix of features, and returns a vector of column vectors representing Aᴸ outputs for each sample, which in this case is only one. Hence why we wrap featuresMatrix[i] in a vector, since were essentially getting predictions on an entire dataset that consists of only one sample)
                */
                std::vector<std::vector<double>> A_L = getPredictions({featuresMatrix[i]})[0];
                printDebug("Network Predictions");
                printMatrixDebug(A_L);

                printDebug("--------------------Backward Pass-------------------------");
                /* 
                The loss for this one sample is the squarred error / 2 (to make the derivative easier) essentially takes the column vector of predictions, Aᴸ, and subtracts the labels vector Y, to produce another vector representing the error of each node output aᵢᴸ with respect to each label yᵢ. We then take the dot product of this resultant vector with itself to get the sum of squarred errors for each term, and divide it by 2 this will represent the loss for this samples forward pass
                */
                double loss = calculateLoss(A_L, Y);
                printDebug("Loss for Sample");
                printDebug(loss);
                epoch_accumulated_loss += loss;

                /*
                For this one sample, we want to find the gradient of the cost/loss function with respect to each layers' weights starting with the last hidden layer L
                so we begin by finding the partial derivative of the cost function with respect
                to the weights of the last hidden layer L, ∂Cᵢ/∂Wᴸ
                */

               /*
               ∂Cᵢ/∂Wᴸ = δᴸ * (Aˡ⁻¹)ᵀ, where * is a outer product
               we will begin by finding δᴸ
               */
                std::vector<std::vector<double>> pd_Ci_W_L;

                /*
                calculte the error term, δᴸ term for the last hidden layer L and remember that derivative of the modified squarred error is just the difference of the two vectors
                δᴸ = g'(Zᴸ) ⊙ (Aᴸ-Y)
                */
                int outputLayer_index = num_hidden_layers - 1;
                
                printDebug("g'(Zᴸ)");
                printMatrixDebug(this->layers[outputLayer_index]->getDerivativeActivationOutputs());

                printDebug("(Aᴸ-Y)");
                printMatrixDebug(subtractColumnVectors(A_L, Y));

                std::vector<std::vector<double>> error_term_L = hadamardProduct(
                    this->layers[outputLayer_index]->getDerivativeActivationOutputs(),
                    subtractColumnVectors(A_L, Y));
                printDebug("δᴸ");
                printMatrixDebug(error_term_L);

                // including the output layer as a hidden layer
                int prev_hidden_layer_index = num_hidden_layers-2;


                // Aˡ⁻¹ but tacking on a 1 to the bottom of the column vector for the bias term
                std::vector<std::vector<double>> A_prev_activation_with_bias;
                
                // if there are other hidden layers
                if (prev_hidden_layer_index >= 0) {
                    // get the previous layers activations
                    A_prev_activation_with_bias = this->layers[prev_hidden_layer_index]->getActivationOutputs();
                } else{
                    // this would just be 1 input layer, 1 output layer NN
                    // otherwise we just retreive the input layers activations which were just the inputs into the network
                    A_prev_activation_with_bias = this->inputLayer->getInputs();
                }
                A_prev_activation_with_bias.push_back({1}); 

                printDebug("Previous Layer Output With Bias");
                printMatrixDebug(A_prev_activation_with_bias);

                /*
                ∂Cᵢ/∂Wᴸ = δᴸ * (Aᴸ⁻¹)ᵀ
                */
                pd_Ci_W_L = outerProduct(error_term_L, takeTranspose(A_prev_activation_with_bias));
                printDebug("∂Cᵢ/∂Wᴸ");
                printMatrixDebug(pd_Ci_W_L);

                // push the partial derivative of the cost with respect to this layers weights to the gradient for this particular sample
                gradient_Ci.push_back(pd_Ci_W_L);
                

                // Propogate gradients to previous layers starting with the layer L-1 stopping and stopping at the input layer
                std::vector<std::vector<double>> prev_error_term = error_term_L;


                // for every hidden layer
                // if this is negative this means that there arent layers to propogate through (i.e a two layer network)
                for (int j = prev_hidden_layer_index; j >= 0; j--) {
                    /*
                    For any hidden layer l, 
                    ∂Cᵢ/∂Wˡ = δˡ * (Aˡ⁻¹)ᵀ, where * is an outer product and 
                    δˡ = ((Wˡ⁺¹)ᵀ * δˡ⁺¹) ⊙ g'(Zˡ)
                    */
                    std::vector<std::vector<double>> pd_Ci_W_l;
                    std::vector<std::vector<double>> error_term;
                    std::vector<std::vector<double>> W_l_plus_1_T;

                    W_l_plus_1_T = takeTranspose(this->layers[j+1]->getWeightsMatrix());

                    printDebug("(Wˡ⁺¹)ᵀ");
                    printMatrixDebug(W_l_plus_1_T);

                    std::vector<std::vector<double>> error_term_lhs = matrixMultiply(W_l_plus_1_T, prev_error_term);

                    error_term_lhs.pop_back(); // remove the bias contribution


                    std::vector<std::vector<double>> DZ_hidden_with_no_bias = this->layers[j]->getDerivativeActivationOutputs();

                    
                    error_term = hadamardProduct(error_term_lhs, DZ_hidden_with_no_bias);

                    printDebug("δˡ");
                    printMatrixDebug(error_term);

                    if (j == 0) { // need to get input layer outputs
                        A_prev_activation_with_bias = this->inputLayer->getInputs();
                    } else {
                        A_prev_activation_with_bias = this->layers[j-1]->getActivationOutputs();
                        
                    }

                    A_prev_activation_with_bias.push_back({1}); 

                    std::vector<std::vector<double>> A_prev_activation_with_bias_T = takeTranspose(A_prev_activation_with_bias);

                    
                    pd_Ci_W_l = outerProduct(error_term, A_prev_activation_with_bias_T);
                    printDebug("∂Cᵢ/∂Wˡ");
                    printMatrixDebug(pd_Ci_W_l);

                    gradient_Ci.push_back(pd_Ci_W_l);
                    prev_error_term = error_term;
                }


                // update weights for every hidden layer using the gradient for that layers weights
                for (int m = 0; m < num_hidden_layers; m++) {
                int gradient_index = gradient_Ci.size() - 1 - m;
                this->layers[m]->updateNeuronWeights(gradient_Ci[gradient_index], this->LR);
                }

                // norm of gradient for this particular sample
                double gradient_sum_of_sub_gradients = 0;
                // for every gradient matrix in the vector of gradients
                for (int i = 0; i < gradient_Ci.size(); i++) {
                    gradient_sum_of_sub_gradients += calculateMatrixEuclideanNorm(gradient_Ci[i]);
                }
                epoch_accumulated_gradient += gradient_sum_of_sub_gradients;

                printDebug("------------------------END EPOCH-------------------------------");
                
            }
            // mean squarred error for this epoch
            double epoch_Loss = epoch_accumulated_loss / num_samples; 
            this->epoch_losses.push_back(epoch_Loss);
            // calcuate the average gradient over this epoch as well
            double average_gradient = epoch_accumulated_gradient / num_samples;
            this->epoch_gradient_norms.push_back(average_gradient);

            std::cout << "Epoch: " << e << " Loss: " << epoch_Loss << " | Average Gradient: " << average_gradient << std::endl;
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
            accumulated_final_model_loss += calculateLoss(best_predictions[i], labels_as_col_vectors[i]);
        }
        this->model_loss = accumulated_final_model_loss / labels.size();
    }

    // currently this returns a vector of column vectors representing the outputs for each forward pass of each sample
    std::vector<std::vector<std::vector<double>>> getPredictions(std::vector<std::vector<double>> featuresMatrix) {

        int num_samples = featuresMatrix.size();

        std::vector<std::vector<std::vector<double>>> predictions;

        // for every sample
        for (int i = 0; i < num_samples; i++) {
            // pass the sample into the input layer and get output
            this->inputLayer->storeInputs(vector1DtoColumnVector(featuresMatrix[i]));
            std::vector<std::vector<double>> input_layer_output = this->inputLayer->getInputs();
            printDebug("------------------------------------------------Getting Predictions--------------------------------------------------------------");
            printDebug("Input Layer Output");
            printMatrixDebug(input_layer_output);
            // for every hidden layer in the network
            std::vector<std::vector<double>> prev_layer_output = input_layer_output;
            for (int j = 0; j < num_hidden_layers; j++) {

                this->layers[j]->calculateLayerOutputs(prev_layer_output);
                
                // print("Hidden layer activation outputs");
                // printVector(this->hiddenLayers[j]->activation_outputs);
                std::vector<std::vector<double>> this_layer_output = this->layers[j]->getActivationOutputs();
                printDebug("This Layer Output");
                printMatrixDebug(this_layer_output);
                prev_layer_output = this_layer_output;
            }


            predictions.push_back(this->layers[num_hidden_layers-1]->getActivationOutputs());
            printDebug("---------------------Got Prediction-------------------------------");
        }
        
        return predictions;
    }

    double calculateLoss(const std::vector<std::vector<double>>& predictions, const std::vector<std::vector<double>>&  labels) {
        if (this->selectedLoss == SQUARRED_ERROR) {
            return modifiedSquarredError(predictions, labels);
        } else if (this->selectedLoss == BINARY_CROSS_ENTROPY) {
            return vectorizedLogLoss(predictions, labels);
        } else {
            throw std::invalid_argument("NO LOSS FUNCTION SELECTED");
        }
    }

    void addLayer(std::shared_ptr<Layer> layer) {
        this->layers.push_back(layer);
        this->num_hidden_layers += 1;
    }

    void addInputLayer(std::shared_ptr<InputLayer> inputLayer) {
        this->inputLayer = inputLayer;
    }

};