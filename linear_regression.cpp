#include <vector>
#include <numeric>
#include <iostream>
#include "utility.h"

class LinearRegression {
    public:
    float b = 0;
    float learning_rate;
    std::vector<float> parameters;
    std::vector<float> predictions;

    LinearRegression(float learningRate) {
        this->learning_rate = learningRate;
    }

    std::vector<float> f(std::vector<std::vector<float>> featuresMatrix, std::vector<float> parameters, float b) {
        std::vector<std::vector<float>> features_transpose = takeTranspose(featuresMatrix);
        std::vector<float> predictions;
        for (int i = 0; i < features_transpose.size(); i++) {
            float dot_product = std::inner_product(features_transpose[i].begin(), features_transpose[i].end(), parameters.begin(), 0.0f);
            predictions.push_back(dot_product + b);
        }

        return predictions;
    }
    
    float calculateLoss(std::vector<std::vector<float>> featuresMatrix, std::vector<float> parameters, float b, std::vector<float> labels) {
        std::vector<float> predictions = f(featuresMatrix, parameters, b);
        float MSE = calculateMSE(labels, predictions);
        return MSE;
    }

    void fit(std::vector<std::vector<float>> featuresMatrix, std::vector<float> labels) {
        for (int i = 0; i < featuresMatrix.size(); i++) {
            this->parameters.push_back(0);
        }

        float h = 0.00001;
        bool optimized = false;

        while (optimized != true) {
            std::vector<float> partialDerivatives(parameters.size());

            for (int i = 0; i < parameters.size(); i++) {
                std::vector<float> parameter_plus_h = parameters;
                parameter_plus_h[i] += h;
                float loss_plus_h = calculateLoss(featuresMatrix, parameter_plus_h, b, labels);
                float loss_constant = calculateLoss(featuresMatrix, parameters, b, labels);
                float partial_derivative =(loss_plus_h - loss_constant) / h;

                partialDerivatives[i] = partial_derivative;
                this->parameters[i] -= (learning_rate * partial_derivative);
            }

            float loss_b_plus_h = calculateLoss(featuresMatrix, parameters, b+h, labels);
            float loss_b_constant = calculateLoss(featuresMatrix, parameters, b, labels);
            float b_partial_derivative = (loss_b_plus_h - loss_b_constant) / h;

            this->b -= (learning_rate * b_partial_derivative);

            std::vector<float> new_predictions = f(featuresMatrix, parameters, b);

            float new_loss = calculateLoss(featuresMatrix, parameters, b, labels);
            std::cout << new_loss << std::endl;

            for (int i = 0; i < partialDerivatives.size(); i++) {
                optimized = true;
                if (abs(partialDerivatives[i]) > 0.01 or b_partial_derivative > 0.01) {
                    optimized = false;
                    break;
                }
            }
        }
    }

    std::vector<float> getPredictions(std::vector<std::vector<float>> featuresMatrix) {
        return f(featuresMatrix, parameters, b);
    }
};

int main() {
    std::vector<std::vector<float>> features_matrix = {{1, 2, 3, 4, 5, 6, 7, 8, 9, 10}, 
                                                        {2, 3, 4, 5, 6, 7, 8, 9, 10, 11}};
    std::vector<float> labels = {12.0f, 17.0f, 22.0f, 27.0f, 32.0f, 37.0f, 42.0f, 47.0f, 52.0f, 57.0f};
    LinearRegression LR(0.0001);
    LR.fit(features_matrix, labels);
    

    std::vector<float> toPrint = LR.getPredictions(features_matrix);;

    for (float e : toPrint) {
        std::cout << e << std::endl;
    }

    return 0;
}

