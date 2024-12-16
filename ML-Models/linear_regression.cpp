#include <vector>
#include <numeric>
#include <iostream>
#include "LinearAlgebra.h"
#include "GenFunctions.h"

class LinearRegression {
    public:
    float b = 0;
    float learning_rate;
    std::vector<float> parameters;

    LinearRegression() {
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
        float MSE = calculateMSE(predictions, labels);
        return MSE;
    }

    void fit(std::vector<std::vector<float>> featuresMatrix, std::vector<float> labels) {
        std::vector<std::vector<float>> featuresMatrix_T = takeTranspose(featuresMatrix);

        std::vector<std::vector<float>> A_T_A = matrixMultiply(featuresMatrix_T, featuresMatrix);
        std::vector<float> A_T_b = matrixToVector(matrixMultiply(featuresMatrix_T, vectorToMatrix(labels)));

        std::vector<float> params_including_b = solveSystem(A_T_A, A_T_b);

        this->b = params_including_b[0];

        std::vector<float> params(params_including_b.begin() + 1, params_including_b.end());
        this->parameters = params;

    }

    std::vector<float> getPredictions(std::vector<std::vector<float>> featuresMatrix) {
        return f(featuresMatrix, parameters, b);
    }
};
