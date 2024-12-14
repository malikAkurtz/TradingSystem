#include <iostream>
#include <vector>

void printVector(const std::vector<float>& vec) {
    std::cout << "[ ";
    for (const auto& elem : vec) {
        std::cout << elem << " ";
    }
    std::cout << "]" << std::endl;
}

void printMatrix(const std::vector<std::vector<float>>& matrix) {
    for (const auto& vec : matrix) {
        printVector(vec);
    }
}