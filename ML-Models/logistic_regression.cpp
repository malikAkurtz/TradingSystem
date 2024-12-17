#include <vector>
#include <numeric>
#include <iostream>
#include <cmath>
#include <cfloat>
#include "LinearAlgebra.h"
#include "GenFunctions.h"
#include "Output.h"
#include <random>

class LogisticRegression {
    public:
    double b;
    double learning_rate;
    double loss;
    std::vector<double> parameters;

    LogisticRegression(double learningRate) {
        this->learning_rate = learningRate;
    }

    std::vector<double> f(std::vector<std::vector<double>> featuresMatrix, std::vector<double> parameters, double b) {
        std::vector<double> predictions;

        // for every sample
        for (int i = 0; i < featuresMatrix.size(); i++) {
            double dot_product = innerProduct(featuresMatrix[i], parameters);
            double log_odds = dot_product + b;
            predictions.push_back(sigmoid(log_odds));
        }

        return predictions;
    }
    
    double calculateLoss(std::vector<std::vector<double>> featuresMatrix, std::vector<double> parameters, double b, std::vector<double> labels) {
        std::vector<double> predictions = f(featuresMatrix, parameters, b);
        double logLoss = calculateLogLoss(predictions, labels);
        return logLoss;
    }

    void fit(std::vector<std::vector<double>> featuresMatrix, std::vector<double> labels) {
        std::random_device rd;                         // Non-deterministic random source
        std::mt19937 gen(rd());                        // Seed the generator
        std::uniform_real_distribution<double> dist(0.0, 1.0);

        int num_columns = featuresMatrix[0].size();
        // initialize parameters
        this->parameters.clear();
        this->b = dist(gen);
        for (int i = 0; i < num_columns; i++) {
            this->parameters.push_back(dist(gen));
        }

        int m = featuresMatrix.size(); // aka number of rows aka number of samples
        std::vector<std::vector<double>> featuresMatrix_T = takeTranspose(featuresMatrix);

        bool optimized = false;
        while (!optimized) {

            // parameters
            std::vector<double> gradientParameters;

            std::vector<double> soft_predictions = f(featuresMatrix, this->parameters, this->b);
            // print("Soft Predictions");
            // printVector(soft_predictions);
            std::vector<double> error_vector = subtractVectors(soft_predictions, labels);
            std::vector<std::vector<double>> error_vector_as_matrix = vectorToMatrix(error_vector);
            gradientParameters = matrixToVector(matrixMultiply(featuresMatrix_T, error_vector_as_matrix));
            gradientParameters = divideVector(gradientParameters, m);

            // bias
            double gradientBias = accumulateVector(subtractVectors(soft_predictions, labels)) / m;


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

            double gradientNorm = calculateNorm(gradientParameters);
            double biasNorm = std::abs(gradientBias);

            if (gradientNorm < 0.005 && biasNorm < 0.005) {
                optimized = true;
            }

        }
        
    }

    std::vector<double> getPredictions(std::vector<std::vector<double>> featuresMatrix) {
        return thresholdFunction(f(featuresMatrix, parameters, b), 0.5);
    }
};

