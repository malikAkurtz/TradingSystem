#include <vector>
#include <numeric>
#include "LinearAlgebra.h"
#include "GenFunctions.h"

class LinearRegression {
    public:
    std::vector<float> parameters;
    float loss;

    LinearRegression() {
    }

    std::vector<float> f(std::vector<std::vector<float>> featuresMatrix, std::vector<float> parameters) {
        std::vector<std::vector<float>> matrix_with_ones = featuresMatrix;
        addOnesToFront(matrix_with_ones);
        std::vector<float> predictions;

        for (int i = 0; i < matrix_with_ones.size(); i++) {
            float dot_product = innerProduct(matrix_with_ones[i], parameters);
            predictions.push_back(dot_product);
        }
        return predictions;
    }
    
    float calculateLoss(std::vector<std::vector<float>> featuresMatrix, std::vector<float> parameters, std::vector<float> labels) {
        std::vector<float> predictions = f(featuresMatrix, parameters);
        float MSE = calculateMSE(predictions, labels);
        return MSE;
    }

    void fit(std::vector<std::vector<float>> featuresMatrix, std::vector<float> labels) {
        std::vector<std::vector<float>> matrix_with_ones = featuresMatrix;
        addOnesToFront(matrix_with_ones);
        std::vector<std::vector<float>> featuresMatrix_T = takeTranspose(matrix_with_ones);

        std::vector<std::vector<float>> A_T_A = matrixMultiply(featuresMatrix_T, matrix_with_ones);
        std::vector<float> A_T_b = matrixToVector(matrixMultiply(featuresMatrix_T, vectorToMatrix(labels)));

        this->parameters = solveSystem(A_T_A, A_T_b);

        this->loss = calculateLoss(featuresMatrix, this->parameters, labels);


    }

    std::vector<float> getPredictions(std::vector<std::vector<float>> featuresMatrix) {
        return f(featuresMatrix, parameters);
    }
};
