#include <vector>
#include <numeric>
#include <iostream>
#include <cmath>
#include <cfloat>
#include "LinearAlgebra.h"
#include "GenFunctions.h"

class LogisticRegression {
    public:
    float b = 0;
    float learning_rate;
    std::vector<float> parameters;

    LogisticRegression(float learningRate) {
        this->learning_rate = learningRate;
    }

    std::vector<float> f(std::vector<std::vector<float>> featuresMatrix, std::vector<float> parameters, float b) {
        std::vector<std::vector<float>> features_transpose = takeTranspose(featuresMatrix);
        std::vector<float> predictions;

        for (int i = 0; i < features_transpose.size(); i++) {
            float dot_product = std::inner_product(features_transpose[i].begin(), features_transpose[i].end(), parameters.begin(), 0.0f);
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
        for (int i = 0; i < featuresMatrix.size(); i++) {
            this->parameters.push_back(-1);
        }

        float h = 1e-6;
        bool optimized = false;

        while (optimized != true) {
            std::vector<float> partialDerivatives(parameters.size());

            for (int i = 0; i < parameters.size(); i++) {
                std::vector<float> parameter_plus_h = parameters;
                std::vector<float> parameter_minus_h = parameters;
                parameter_plus_h[i] += h;
                parameter_plus_h[i] -= h;
                float loss_plus_h = calculateLoss(featuresMatrix, parameter_plus_h, b, labels);
                float loss_minus_h = calculateLoss(featuresMatrix, parameter_minus_h, b, labels);
                float partial_derivative =(loss_plus_h - loss_minus_h) / (2*h);

                partialDerivatives[i] = partial_derivative;
                this->parameters[i] -= (learning_rate * partial_derivative);
            }

            float loss_b_plus_h = calculateLoss(featuresMatrix, parameters, b+h, labels);
            float loss_b_minus_h = calculateLoss(featuresMatrix, parameters, b-h, labels);
            float b_partial_derivative = (loss_b_plus_h - loss_b_minus_h) /(2*h);

            this->b -= (learning_rate * b_partial_derivative);

            std::vector<float> new_predictions = f(featuresMatrix, parameters, b);

            float new_loss = calculateLoss(featuresMatrix, parameters, b, labels);
            std::cout << new_loss << std::endl;

            for (int i = 0; i < partialDerivatives.size(); i++) {
                optimized = true;
                if (abs(partialDerivatives[i]) > 0.001 or b_partial_derivative > 0.001) {
                    optimized = false;
                    break;
                }
            }
        }
    }

    std::vector<bool> getPredictions(std::vector<std::vector<float>> featuresMatrix) {
        return thresholdFunction(f(featuresMatrix, parameters, b), 0.5);
    }
};

