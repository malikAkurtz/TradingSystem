#ifndef OUTPUT
#define OUTPUT

#include <vector>
#include <iostream>

extern bool DEBUG;


void printMatrixDebug(const std::vector<std::vector<double>>& matrix);
void printMatrix(const std::vector<std::vector<double>> &matrix);

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


#endif