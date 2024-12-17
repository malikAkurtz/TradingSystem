#include <iostream>
#include <vector>
#include <string>

void printVector(const std::vector<double>& vec) {
    std::cout << "[ ";
    for (const auto& elem : vec) {
        std::cout << elem << " ";
    }
    std::cout << "]" << std::endl;
}

void printMatrix(const std::vector<std::vector<double>>& matrix) {
    std::cout << "------------------------------------------------------------------------------------------------------" << std::endl;
    for (const auto& vec : matrix) {
        printVector(vec);
    }
    std::cout << "------------------------------------------------------------------------------------------------------" << std::endl;
}

void print(std::string toPrint) {
    std::cout << toPrint << std::endl;
}