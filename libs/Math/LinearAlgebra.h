#ifndef LIN_ALGEBRA
#define LIN_ALGEBRA

#include <vector>
#include <cmath>

std::vector<std::vector<float>> takeTranspose(std::vector<std::vector<float>> inputMatrix);
std::vector<float> addVectors(std::vector<float> v1, std::vector<float> v2);
std::vector<float> subtractVectors(std::vector<float> v1, std::vector<float> v2);
float calculateNorm(std::vector<float> v1);
float innerProduct(std::vector<float> v1, std::vector<float> v2);
std::vector<std::vector<float>> matrixMultiply(std::vector<std::vector<float>> m1, std::vector<std::vector<float>> m2);


#endif