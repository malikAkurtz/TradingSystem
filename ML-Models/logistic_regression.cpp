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
    float b = 1;
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
            predictions.push_back(sigmoid(log_odds));
        }

        return predictions;
    }
    
    float calculateLoss(std::vector<std::vector<float>> featuresMatrix, std::vector<float> parameters, float b, std::vector<float> labels) {
        std::vector<float> predictions = f(featuresMatrix, parameters, b);
        float logLoss = calculateLogLoss(predictions, labels);
        return logLoss;
    }

    void fit(std::vector<std::vector<float>> featuresMatrix, std::vector<float> labels) {
        int num_columns = featuresMatrix[0].size();
        // initialize parameters
        this->parameters.clear();
        for (int i = 0; i < num_columns; i++) {
            this->parameters.push_back(1);
        }
        int m = featuresMatrix.size(); // aka number of rows aka number of samples
        std::vector<std::vector<float>> featuresMatrix_T = takeTranspose(featuresMatrix);

        bool optimized = false;
        while (!optimized) {

            // parameters
            std::vector<float> gradientParameters;

            std::vector<float> soft_predictions = f(featuresMatrix, this->parameters, this->b);
            // print("Soft Predictions");
            // printVector(soft_predictions);
            std::vector<float> error_vector = subtractVectors(soft_predictions, labels);
            std::vector<std::vector<float>> error_vector_as_matrix = vectorToMatrix(error_vector);
            gradientParameters = matrixToVector(matrixMultiply(featuresMatrix_T, error_vector_as_matrix));
            gradientParameters = divideVector(gradientParameters, m);

            // bias
            float gradientBias = accumulateVector(subtractVectors(soft_predictions, labels)) / m;


            for (int i = 0; i < this->parameters.size(); i++) {
                this->parameters[i] -= (this->learning_rate * gradientParameters[i]);
            }
            this->b -= (this->learning_rate * gradientBias);

            this->loss = calculateLoss(featuresMatrix, this->parameters, this->b, labels);
            // std::cout << "Parameters: ";
            // printVector(this->parameters);
            // std::cout << "bias: " << this->b << std::endl;
            // std::cout << "Gradient for parameters: ";
            // printVector(gradientParameters);
            // std::cout << "Gradient for bias: " << gradientBias << std::endl;
            std::cout << "Loss: " << this->loss << std::endl;

            float gradientNorm = calculateNorm(gradientParameters);
            float biasNorm = std::abs(gradientBias);

            if (gradientNorm < 0.005 && biasNorm < 0.005) {
                optimized = true;
            }

        }
        
    }

    std::vector<float> getPredictions(std::vector<std::vector<float>> featuresMatrix) {
        return thresholdFunction(f(featuresMatrix, parameters, b), 0.5);
    }
};

