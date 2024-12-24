#include "OptimizationMethods.h"

using namespace LinearAlgebra;

namespace OptimizationMethods
{
    void batchGradientDescent(NeuralNetwork &network, const std::vector<std::vector<double>> &featuresMatrix, const std::vector<std::vector<double>> &labels)
    {
        int num_samples = featuresMatrix.size();
        int num_batches = (num_samples + network.batch_size - 1) / network.batch_size;
        

        std::vector<std::vector<std::vector<double>>> batches_of_features = createBatches(featuresMatrix, network.batch_size);
        std::vector<std::vector<std::vector<double>>> batches_of_labels = createBatches(labels, network.batch_size);
        // ΔJ, vector of matrices representing partial derivatives of cost with respect to each of the layers weights for a single batch
        std::vector<std::vector<std::vector<double>>> gradient_J; 
        
        // for every epoc h
        for (int e = 0; e < network.num_epochs; e++) {
            printDebug("--------------------NEW EPOCH--------------------");
            double epoch_accumulated_loss = 0;
            double epoch_accumulated_gradient = 0;
            // for every sample in the data set
            for (int i = 0; i < num_batches; i++) {
                // we are going to find the gradient of the loss of this particular sample with each layers weights
                gradient_J.clear();

                printDebug("-------------------Forward Pass----------------");
                // Store the features for this particular batch as a matrix

                std::vector<std::vector<double>> cur_X_batch_matrix = batches_of_features[i];
                printDebug("cur_X_batch_matrix");
                printMatrixDebug(cur_X_batch_matrix);

                
                int num_samples_in_batch = cur_X_batch_matrix.size();
                printDebug("Number of Samples in Batch");
                printDebug(num_samples_in_batch);

                std::vector<std::vector<double>> cur_Y_batch_matrix = batches_of_labels[i];


                /* 
                The output of the last hidden layer for this sample, Aᴸ, will be the result of indexing the first and only output of the getPredictions method, which takes in a matrix of features, and returns a vector of column vectors representing Aᴸ outputs for each sample, which in this case is only one. Hence why we wrap featuresMatrix[i] in a vector, since were essentially getting predictions on an entire dataset that consists of only one sample)
                */
                std::vector<std::vector<double>> A_L = network.getPredictions(cur_X_batch_matrix);
                // since A_L was tranpose in the prediction process to make it a vector of column vectors
                std::vector<std::vector<double>> cur_Y_batch_matrix_T = takeTranspose(cur_Y_batch_matrix);

                printDebug("Network Predictions");
                printMatrixDebug(A_L);

                printDebug("--------------------Backward Pass-------------------------");
                /* 
                The loss for this one sample is the squarred error / 2 (to make the derivative easier) essentially takes the column vector of predictions, Aᴸ, and subtracts the labels vector Y, to produce another vector representing the error of each node output aᵢᴸ with respect to each label yᵢ. We then take the dot product of this resultant vector with itself to get the sum of squarred errors for each term, and divide it by 2 this will represent the loss for this samples forward pass
                */
                double batch_loss = 0;
                int num_colums_of_A_L = A_L[0].size();


                for (int j = 0; j < num_colums_of_A_L; j++)
                {
                    batch_loss += network.calculateLoss(getColumn(A_L, j), getColumn(cur_Y_batch_matrix_T, j));
                }
                batch_loss = batch_loss / num_samples_in_batch;
                printDebug("Loss for Batch");
                printDebug(batch_loss);
                epoch_accumulated_loss += batch_loss;

                /*
                For this one sample, we want to find the gradient of the cost/loss function with respect to each layers' weights starting with the last hidden layer L
                so we begin by finding the partial derivative of the cost function with respect
                to the weights of the last hidden layer L, ∂Cᵢ/∂Wᴸ
                */

                /*
                ∂J/∂Wᴸ = δᴸ * (Aˡ⁻¹)ᵀ, where * is a outer product and
                δᴸ = g'(Zᴸ) ⊙ (Aᴸ-Y)
                */
                std::vector<std::vector<double>> pd_J_W_L;

                /*
                calculte the error term, δᴸ term for the last hidden layer L and remember that derivative of the modified squarred error is just the difference of the two matrices
                */
                int outputLayer_index = network.num_hidden_layers - 1;
                
                printDebug("g'(Zᴸ)");
                printMatrixDebug(network.layers[outputLayer_index]->getDerivativeActivationOutputs());

                printDebug("(Aᴸ-Y)");
                printDebug("A_L");
                printMatrixDebug(A_L);
                printDebug("cur_Y_batch_matrix");
                printMatrixDebug(cur_Y_batch_matrix_T);
                printMatrixDebug(subtractMatrices(A_L, cur_Y_batch_matrix_T));

                std::vector<std::vector<double>> error_term_L = hadamardProduct(
                    network.layers[outputLayer_index]->getDerivativeActivationOutputs(),
                    subtractMatrices(A_L, cur_Y_batch_matrix_T));
                printDebug("δᴸ");
                printMatrixDebug(error_term_L);

                // since were including the output layer as a hidden layer
                int prev_hidden_layer_index = network.num_hidden_layers-2;


                // Aˡ⁻¹ but tacking on a 1 to the bottom of the matrix for the bias term
                std::vector<std::vector<double>> A_prev_activation_with_bias;
                
                // if there are other hidden layers
                if (prev_hidden_layer_index >= 0) {
                    // get the previous layers activations
                    A_prev_activation_with_bias = network.layers[prev_hidden_layer_index]->getActivationOutputs();
                } else{
                    // this would just be 1 input layer, 1 output layer NN
                    // otherwise we just retreive the input layers activations which were just the inputs into the network
                    A_prev_activation_with_bias = network.inputLayer->getInputs();
                }
                std::vector<double> ones_to_append(A_prev_activation_with_bias[0].size(), 1);

                A_prev_activation_with_bias.push_back(ones_to_append); // Add bias

                printDebug("Previous Layer Output With Bias");
                printMatrixDebug(A_prev_activation_with_bias);

                /*
                ∂Cᵢ/∂Wᴸ = δᴸ * (Aᴸ⁻¹)ᵀ
                */
                pd_J_W_L = matrixMultiply(error_term_L, takeTranspose(A_prev_activation_with_bias));
                printDebug("∂J/∂Wᴸ");
                printMatrixDebug(pd_J_W_L);

                // push the partial derivative of the cost with respect to this layers weights to the gradient for this particular sample
                gradient_J.push_back(pd_J_W_L);
                

                // Propogate gradients to previous layers starting with the layer L-1 stopping and stopping at the input layer
                std::vector<std::vector<double>> prev_error_term = error_term_L;


                // for every hidden layer
                // if this is negative this means that there arent layers to propogate through (i.e a two layer network)
                for (int j = prev_hidden_layer_index; j >= 0; j--) {
                    /*
                    For any hidden layer l, 
                    ∂J/∂Wˡ = δˡ * (Aˡ⁻¹)ᵀ,
                    δˡ = ((Wˡ⁺¹)ᵀ * δˡ⁺¹) ⊙ g'(Zˡ)
                    */
                    std::vector<std::vector<double>> pd_J_W_l;
                    std::vector<std::vector<double>> error_term;
                    std::vector<std::vector<double>> W_l_plus_1_T;

                    W_l_plus_1_T = takeTranspose(network.layers[j+1]->getWeightsMatrix());

                    printDebug("(Wˡ⁺¹)ᵀ");
                    printMatrixDebug(W_l_plus_1_T);

                    std::vector<std::vector<double>> error_term_lhs = matrixMultiply(W_l_plus_1_T, prev_error_term);

                    error_term_lhs.pop_back(); // remove the bias contribution


                    std::vector<std::vector<double>> DZ_hidden_with_no_bias = network.layers[j]->getDerivativeActivationOutputs();

                    
                    error_term = hadamardProduct(error_term_lhs, DZ_hidden_with_no_bias);

                    printDebug("δˡ");
                    printMatrixDebug(error_term);

                    if (j == 0) { // need to get input layer outputs
                        A_prev_activation_with_bias = network.inputLayer->getInputs();
                    } else {
                        A_prev_activation_with_bias = network.layers[j-1]->getActivationOutputs();
                        
                    }

                    std::vector<double> ones_to_append(A_prev_activation_with_bias[0].size(), 1);

                    A_prev_activation_with_bias.push_back(ones_to_append); // Add bias

                    std::vector<std::vector<double>> A_prev_activation_with_bias_T = takeTranspose(A_prev_activation_with_bias);

                    
                    pd_J_W_l = matrixMultiply(error_term, A_prev_activation_with_bias_T);
                    printDebug("∂J/∂Wˡ");
                    printMatrixDebug(pd_J_W_l);

                    gradient_J.push_back(pd_J_W_l);
                    prev_error_term = error_term;
                }


                // update weights for every hidden layer using the gradient for that layers weights
                for (int m = 0; m < network.num_hidden_layers; m++) 
                {
                    int gradient_index = gradient_J.size() - 1 - m;
                    network.layers[m]->updateNeuronWeights(gradient_J[gradient_index], network.LR);
                }

                // norm of gradient for this particular sample
                double gradient_sum_of_sub_gradients = 0;
                // for every gradient matrix in the vector of gradients
                for (int i = 0; i < gradient_J.size(); i++) {
                    gradient_sum_of_sub_gradients += calculateMatrixEuclideanNorm(gradient_J[i]);
                }
                epoch_accumulated_gradient += gradient_sum_of_sub_gradients;

                printDebug("------------------------END EPOCH-------------------------------");
                        }
            // mean squarred error for this epoch
            double epoch_Loss = epoch_accumulated_loss / num_samples; 
            network.epoch_losses.push_back(epoch_Loss);
            // calcuate the average gradient over this epoch as well
            double average_gradient = epoch_accumulated_gradient / num_samples;
            network.epoch_gradient_norms.push_back(average_gradient);

            std::cout << "Epoch: " << e << " Loss: " << epoch_Loss << " | Average Gradient: " << average_gradient << std::endl;
        }
        // now getting predictions of the entire feature matrix, i.e all samples
        // best_predictions will then consist of a vector of column vectors
        std::vector<std::vector<double>> best_predictions = network.getPredictions(featuresMatrix);
        std::vector<std::vector<double>> labels_T = takeTranspose(labels);
        double accumulated_final_model_loss = 0;
        printDebug("best_predictions");
        printMatrixDebug(best_predictions);
        printDebug("labels_T");
        printMatrixDebug(labels_T);
        for (int i = 0; i < best_predictions.size(); i++)
        {
            accumulated_final_model_loss += network.calculateLoss(getColumn(best_predictions, i), getColumn(labels_T, i));
        }
        network.model_loss = accumulated_final_model_loss / labels.size();
    }


    void NeuroEvolution(NeuralNetwork &network, const std::vector<std::vector<double>> &featuresMatrix, const std::vector<std::vector<double>> &labels)
    {
        int num_samples = featuresMatrix.size();
        std::vector<std::vector<double>> labels_T = takeTranspose(labels);
        int population_size = 100;
        int max_generations = 1000;

        // Initialize Population
        std::vector<NeuralNetwork> population(population_size, network);

        for (int i = 0; i < population_size; i++)
        {
            population[i].reInitializeLayers();
        }

        // for every generation
        for (int g = 0; g < max_generations; g++)
        {  
            std::vector<std::pair<double, NeuralNetwork>> population_loss;
            // need to evaluate the population
            // for every network in the population
            for (int i = 0; i < population_size; i++)
            {
                NeuralNetwork &thisNN = population[i];
                // perform a forward pass of the entire dataset through this network

                // A matrix where each column is a prediction for that sample from left to right
                std::vector<std::vector<double>> thisNNoutputs = thisNN.getPredictions(featuresMatrix);

                double thisNNloss = 0;
                for (int j = 0; j < thisNNoutputs.size(); j++)
                {
                    thisNNloss += network.calculateLoss(getColumn(thisNNoutputs, j), getColumn(labels_T, j));
                }
                thisNNloss /= num_samples;
                std::pair<double, NeuralNetwork> NNxLoss(thisNNloss, thisNN);

                population_loss.push_back(NNxLoss);
            }

            // sort the dictionary
            std::sort(population_loss.begin(), population_loss.end());

            // Elitism selection, keeping top 20%
            int num_surviving_networks = population_size * 0.2;

            std::vector<std::pair<double, NeuralNetwork>> elites(population_loss.begin(), population_loss.begin() + num_surviving_networks + 1);

            population.clear();
            for (auto& elite : elites)
            {
                population.push_back(elite.second);
            }

            // crossover/breeding
            std::vector<NeuralNetwork> children(elites.size());


        }
    }
}