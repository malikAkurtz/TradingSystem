#include "ML-Models/linear_regression.cpp"
#include <iostream>
#include "LinearAlgebra.h"
#include "Output.h"
#include <iomanip> // Required for std::fixed and std::setprecision

int main() {
    std::vector<std::vector<float>> A = {
        {1, 1},
        {1, 3},
        {1, 4}
    };

    std::vector<float> b = {3, 7, 9};

    LinearRegression LR;

    LR.fit(A, b);

    // Set output formatting for floating-point numbers
    std::cout << std::fixed << std::setprecision(6);

    // Print results
    std::cout << "Intercept (b): " << LR.b << std::endl;
    printVector(LR.parameters);

    return 0;
}

