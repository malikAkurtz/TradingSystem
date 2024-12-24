#include "Output.h"


using namespace LinearAlgebra;

void printMatrix(const std::vector<std::vector<double>>& matrix) {
    std::cout << "[" << std::endl;
    for (const auto& row : matrix) {
        std::cout << "  < ";
        for (const auto& elem : row) {
            std::cout << elem << " ";
        }
        std::cout << ">" << std::endl;
    }
    std::cout << "]" << std::endl;
}

void printMatrixDebug(const std::vector<std::vector<double>>& matrix) {
    if (DEBUG) {
        std::cout << "[" << std::endl;
        for (const auto& row : matrix) {
            std::cout << "  < ";
            for (const auto& elem : row) {
                std::cout << elem << " ";
            }
            std::cout << ">" << std::endl;
        }
        std::cout << "]" << std::endl;
        std::cout << "(" << matrix.size() << "," << matrix[0].size() << ")" << std::endl;
    }
}



////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////



void printPredictionsVSLabels(const std::vector<std::vector<double>> &predictions, 
                              const std::vector<std::vector<double>> &labels) {
    // Transpose predictions so each row corresponds to a sample
    std::vector<std::vector<double>> predictions_T = takeTranspose(predictions);

    // Check size compatibility
    if (predictions_T.size() != labels.size()) {
        std::cerr << "Error: Predictions and labels sizes do not match!" << std::endl;
        return;
    }

    // Iterate over all prediction-label pairs
    for (size_t i = 0; i < predictions_T.size(); i++) {
        std::cout << "Prediction-Label Pair " << i + 1 << ": ";

        // Check if sizes of predictions and labels for the current sample match
        if (predictions_T[i].size() != labels[i].size()) {
            std::cerr << "Error: Mismatch in the number of outputs for prediction and label at index " << i << "!" << std::endl;
            continue;
        }

        std::cout << "<";
        for (size_t j = 0; j < predictions_T[i].size(); j++) {
            // Print each prediction-label pair for this sample
            std::cout << "(" << predictions_T[i][j] << ", " << labels[i][j] << ")";
            if (j < predictions_T[i].size() - 1) {
                std::cout << ", "; // Add a comma separator if not the last pair
            }
        }
        std::cout << ">" << std::endl;
    }
}
