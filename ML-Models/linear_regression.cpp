#include <vector>
#include <numeric>
#include "LinearAlgebra.h"
#include "GenFunctions.h"

class LinearRegression {
    public:
    std::vector<double> parameters;
    double loss;

    LinearRegression() {
    }

    std::vector<double> f(std::vector<std::vector<double>> featuresMatrix, std::vector<double> parameters) {
        std::vector<std::vector<double>> matrix_with_ones = featuresMatrix;
        addOnesToFront(matrix_with_ones);
        std::vector<double> predictions;

        for (int i = 0; i < matrix_with_ones.size(); i++) {
            double dot_product = innerProduct(matrix_with_ones[i], parameters);
            predictions.push_back(dot_product);
        }
        return predictions;
    }
    
    double calculateLoss(std::vector<std::vector<double>> featuresMatrix, std::vector<double> parameters, std::vector<double> labels) {
        std::vector<double> predictions = f(featuresMatrix, parameters);
        double MSE = calculateMSE(predictions, labels);
        return MSE;
    }

    void fit(std::vector<std::vector<double>> featuresMatrix, std::vector<double> labels) {
        std::vector<std::vector<double>> matrix_with_ones = featuresMatrix;
        addOnesToFront(matrix_with_ones);
        std::vector<std::vector<double>> featuresMatrix_T = takeTranspose(matrix_with_ones);

        std::vector<std::vector<double>> A_T_A = matrixMultiply(featuresMatrix_T, matrix_with_ones);
        std::vector<double> A_T_b = matrixToVector(matrixMultiply(featuresMatrix_T, vectorToMatrix(labels)));

        this->parameters = solveSystem(A_T_A, A_T_b);

        this->loss = calculateLoss(featuresMatrix, this->parameters, labels);


    }

    std::vector<double> getPredictions(std::vector<std::vector<double>> featuresMatrix) {
        return f(featuresMatrix, parameters);
    }
};
