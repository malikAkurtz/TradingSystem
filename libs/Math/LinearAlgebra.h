#ifndef LIN_ALGEBRA
#define LIN_ALGEBRA

#include <vector>
#include <cmath>

std::vector<std::vector<double>> takeTranspose(std::vector<std::vector<double>> inputMatrix);
std::vector<double> addVectors(std::vector<double> v1, std::vector<double> v2);
std::vector<double> subtractVectors(std::vector<double> v1, std::vector<double> v2);
double calculateNorm(std::vector<double> v1);
double innerProduct(std::vector<double> v1, std::vector<double> v2);
std::vector<std::vector<double>> matrixMultiply(std::vector<std::vector<double>> m1, std::vector<std::vector<double>> m2);
void addRow(std::vector<std::vector<double>>& matrix, std::vector<double> row);
void addColumn(std::vector<std::vector<double>>& matrix, std::vector<double> column);
std::vector<double> solveSystem(std::vector<std::vector<double>> matrix, std::vector<double> b);
void deleteColumn(std::vector<std::vector<double>>& matrix, int column_index);
void deleteRow(std::vector<std::vector<double>>& matrix, int row_index);
std::vector<std::vector<double>> vectorToMatrix(std::vector<double> vector);
std::vector<double> matrixToVector(std::vector<std::vector<double>> matrix);
std::vector<double> scaleVector(std::vector<double> v1, double scalar);
void addOnesToFront(std::vector<std::vector<double>>& matrix);
std::vector<double> getColumn(std::vector<std::vector<double>> matrix, int col_index);
std::vector<double> getRow(std::vector<std::vector<double>> matrix, int row_index);
double accumulateVector(std::vector<double> v1);
void updateColumn(std::vector<std::vector<double>>& matrix, std::vector<double> v1, int col_index);
std::vector<double> createVector(double num, int length);
std::vector<double> divideVector(std::vector<double> v1, double scalar);


#endif