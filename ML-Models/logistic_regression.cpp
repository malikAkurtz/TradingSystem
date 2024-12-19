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
    double learning_rate;
    double loss;
    std::vector<double> parameters;

    LogisticRegression(double learningRate) {
        this->learning_rate = learningRate;
    }

    std::vector<double> f(std::vector<std::vector<double>> featuresMatrix, std::vector<double> parameters) {
        std::vector<std::vector<double>> matrix_with_ones = featuresMatrix;
        addOnesToFront(matrix_with_ones);
        std::vector<double> predictions;
        
        // for every sample
        for (int i = 0; i < matrix_with_ones.size(); i++) {
            double dot_product = innerProduct(matrix_with_ones[i], parameters);
            double log_odds = dot_product;
            predictions.push_back(sigmoid_single(log_odds)); // can vectorize this but not now
        }

        return predictions;
    }
    
    double calculateLoss(std::vector<std::vector<double>> featuresMatrix, std::vector<double> parameters, std::vector<double> labels) {
        std::vector<double> predictions = f(featuresMatrix, parameters);
        double logLoss = calculateLogLoss(predictions, labels);
        return logLoss;
    }

    void fit(std::vector<std::vector<double>> featuresMatrix, std::vector<double> labels) {
        std::random_device rd;                         // Non-deterministic random source
        std::mt19937 gen(rd());                        // Seed the generator
        std::uniform_real_distribution<double> dist(0.0, 1.0);

        std::vector<std::vector<double>> matrix_with_ones = featuresMatrix;
        addOnesToFront(matrix_with_ones);

        int num_columns = matrix_with_ones[0].size();
        // initialize parameters
        this->parameters.clear();
        for (int i = 0; i < num_columns; i++) {
            this->parameters.push_back(dist(gen));
        }

        int m = matrix_with_ones.size(); // aka number of rows aka number of samples
        std::vector<std::vector<double>> featuresMatrix_T = takeTranspose(matrix_with_ones);

        bool optimized = false;
        while (!optimized) {

            // parameters
            std::vector<double> gradientParameters;

            std::vector<double> soft_predictions = f(featuresMatrix, this->parameters);
            // print("Soft Predictions");
            // printVector(soft_predictions);
            std::vector<double> error_vector = subtractVectors(soft_predictions, labels);
            std::vector<std::vector<double>> error_vector_as_matrix = vector1Dto2D(error_vector);
            gradientParameters = vector2Dto1D(matrixMultiply(featuresMatrix_T, error_vector_as_matrix));
            gradientParameters = divideVector(gradientParameters, m);

            for (int i = 0; i < this->parameters.size(); i++) {
                this->parameters[i] -= (this->learning_rate * gradientParameters[i]);
            }

            this->loss = calculateLoss(featuresMatrix, this->parameters, labels);
            // std::cout << "Parameters: ";
            // printVector(this->parameters);
            // std::cout << "bias: " << this->b << std::endl;
            // std::cout << "Gradient for parameters: ";
            // printVector(gradientParameters);
            // std::cout << "Gradient for bias: " << gradientBias << std::endl;
            std::cout << "Loss: " << this->loss << std::endl;

            double gradientNorm = calculateNorm(gradientParameters);

            if (gradientNorm < 0.005) {
                optimized = true;
            }

        }
        
    }

    std::vector<double> getPredictions(std::vector<std::vector<double>> featuresMatrix) {
        return thresholdFunction(f(featuresMatrix, parameters), 0.5);
    }
};

