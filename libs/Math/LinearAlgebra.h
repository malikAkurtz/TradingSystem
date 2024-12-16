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
void addRow(std::vector<std::vector<float>>& matrix, std::vector<float> row);
void addColumn(std::vector<std::vector<float>>& matrix, std::vector<float> column);
std::vector<float> solveSystem(std::vector<std::vector<float>> matrix, std::vector<float> b);
void deleteColumn(std::vector<std::vector<float>>& matrix, int column_index);
void deleteRow(std::vector<std::vector<float>>& matrix, int row_index);
std::vector<std::vector<float>> vectorToMatrix(std::vector<float> vector);
std::vector<float> matrixToVector(std::vector<std::vector<float>> matrix);


#endif