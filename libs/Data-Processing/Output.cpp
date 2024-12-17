#include <iostream>
#include <vector>

void printVector(const std::vector<float>& vec) {
    std::cout << "------------------------------------------------------------------------------------------------------" << std::endl;
    std::cout << "[ ";
    for (const auto& elem : vec) {
        std::cout << elem << " ";
    }
    std::cout << "]" << std::endl;
    std::cout << "------------------------------------------------------------------------------------------------------" << std::endl;
}

void printMatrix(const std::vector<std::vector<float>>& matrix) {
    std::cout << "------------------------------------------------------------------------------------------------------" << std::endl;
    for (const auto& vec : matrix) {
        printVector(vec);
    }
    std::cout << "------------------------------------------------------------------------------------------------------" << std::endl;
}