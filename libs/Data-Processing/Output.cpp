#include <iostream>
#include <vector>
#include <string>

void printVector(const std::vector<double>& vec) {
    std::cout << "< ";
    for (const auto& elem : vec) {
        std::cout << elem << " ";
    }
    std::cout << ">" << std::endl;
}

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

void print(std::string toPrint) {
    std::cout << toPrint << std::endl;
}

void printPredictionsVSLabels(std::vector<double> predictions, std::vector<double> labels) {
    for (int i = 0; i < predictions.size(); i++) {
        std::cout << "<" << predictions[i] << ", " << labels[i] << ">" << std::endl;
    }
}

void printMatrixShape(std::vector<std::vector<double>> matrix) {
    std::cout << "(" << matrix.size() << "," << matrix[0].size() << ")" << std::endl;
}

void printVectorShape(std::vector<double> vector) {
    std::cout << "(" << vector.size() << ", " << ")" << std::endl;
}