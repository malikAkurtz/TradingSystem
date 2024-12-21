#ifndef OUTPUT
#define OUTPUT

#include <vector>
#include <iostream>

extern bool DEBUG;

void printVectorDebug(const std::vector<double>& vec);
void printVector(const std::vector<double>& vec);


void printMatrixDebug(const std::vector<std::vector<double>>& matrix);
void printMatrix(const std::vector<std::vector<double>>& matrix);

template <typename T>
void print(const T& value) {
    std::cout << value << std::endl;
}

template <typename T>
void printDebug(const T& value) {
    if (DEBUG) {
        std::cout << value << std::endl;
    }
}

void printPredictionsVSLabels(const std::vector<std::vector<std::vector<double>>>& predictions, 
                              const std::vector<std::vector<double>>& labels);

void printMatrixShape(const std::vector<std::vector<double>>& matrix);
void printMatrixShapeDebug(const std::vector<std::vector<double>>& matrix);

void printVectorShapeDebug(const std::vector<double>& vector);
void printVectorShape(const std::vector<double>& vector);

#endif