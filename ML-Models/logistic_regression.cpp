#include <vector>
#include <numeric>
#include <iostream>
#include <cmath>
#include <cfloat>
#include "LinearAlgebra.h"
#include "GenFunctions.h"
#include "Output.h"

class LogisticRegression {
    public:
    float b = 0;
    float learning_rate;
    float loss;
    std::vector<float> parameters;

    LogisticRegression(float learningRate) {
        this->learning_rate = learningRate;
    }

    std::vector<float> f(std::vector<std::vector<float>> featuresMatrix, std::vector<float> parameters, float b) {
        std::vector<float> predictions;

        // for every sample
        for (int i = 0; i < featuresMatrix.size(); i++) {
            float dot_product = innerProduct(featuresMatrix[i], parameters);
            float log_odds = dot_product + b;
            predictions.push_back(exp(log_odds) / (1 + exp(log_odds)));
        }

        return predictions;
    }
    
    float calculateLoss(std::vector<std::vector<float>> featuresMatrix, std::vector<float> parameters, float b, std::vector<float> labels) {
        std::vector<float> predictions = f(featuresMatrix, parameters, b);
        float logLoss = calculateLogLoss(predictions, labels);
        return logLoss;
    }

    void fit(std::vector<std::vector<float>> featuresMatrix, std::vector<float> labels) {
        for (int i = 0; i < featuresMatrix[0].size(); i++) {
            this->parameters.push_back(1);
        }

        float h = 1e-4;
        bool optimized = false;

        while (optimized != true) {
            std::vector<float> partialDerivatives(parameters.size());

            for (int i = 0; i < parameters.size(); i++) {
                std::vector<float> parameter_plus_h = this->parameters;
                std::vector<float> parameter_minus_h = this->parameters;
                parameter_plus_h[i] += h;
                parameter_minus_h[i] -= h;
                float loss_plus_h = calculateLoss(featuresMatrix, parameter_plus_h, b, labels);
                float loss_minus_h = calculateLoss(featuresMatrix, parameter_minus_h, b, labels);
                float partial_derivative =(loss_plus_h - loss_minus_h) / (2*h);


                partialDerivatives[i] = partial_derivative;
                
            }

            float loss_b_plus_h = calculateLoss(featuresMatrix, this->parameters, this->b+h, labels);
            float loss_b_minus_h = calculateLoss(featuresMatrix, this->parameters, this->b-h, labels);
            float b_partial_derivative = (loss_b_plus_h - loss_b_minus_h) /(2*h);

            for (int i = 0; i < this->parameters.size(); i++) {
                this->parameters[i] -= (learning_rate * partialDerivatives[i]);
            }
            this->b -= (learning_rate * b_partial_derivative);


            std::vector<float> new_predictions = f(featuresMatrix, this->parameters, b);
            float new_loss = calculateLoss(featuresMatrix, this->parameters, b, labels);
            this->loss = new_loss;


            // printVector(this->parameters);
            std::cout << this->loss << std::endl;
            // printVector(partialDerivatives);

            float gradient_norm = 0.0;

            for (float grad : partialDerivatives) {
                gradient_norm += grad * grad;
            }
            

            gradient_norm = std::sqrt(gradient_norm);

            optimized = (gradient_norm < 1e-2 && std::abs(b_partial_derivative) < 1e-2);
            std::cout << "Partial Derivatives: " << std::endl;
            printVector(partialDerivatives);
        }
        
    }

    std::vector<float> getPredictions(std::vector<std::vector<float>> featuresMatrix) {
        return thresholdFunction(f(featuresMatrix, parameters, b), 0.5);
    }
};

