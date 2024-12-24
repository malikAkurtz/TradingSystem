#ifndef LIN_ALGEBRA
#define LIN_ALGEBRA

#include <vector>
#include <cmath>
#include <stdexcept>
#include <Output.h>
 namespace LinearAlgebra 
 {
    std::vector<std::vector<double>> takeTranspose(const std::vector<std::vector<double>>& inputMatrix);
    std::vector<double> addVectors(const std::vector<double>& v1, const std::vector<double>& v2);
    std::vector<double> subtractVectors(const std::vector<double>& v1, const std::vector<double>& v2);
    double calculateNorm(const std::vector<double>& v1);
    double innerProduct(const std::vector<double>& v1, const std::vector<double>& v2);
    std::vector<std::vector<double>> matrixMultiply(const std::vector<std::vector<double>>& m1, const std::vector<std::vector<double>>& m2);
    void addRow(std::vector<std::vector<double>>& matrix, std::vector<double> row);
    void addColumn(std::vector<std::vector<double>>& matrix, std::vector<double> column);
    std::vector<std::vector<double>> createColumnVector(const double &num, const int &length);
    std::vector<double> solveSystem(const std::vector<std::vector<double>> &matrix, const std::vector<double> &b);
    void deleteColumn(std::vector<std::vector<double>>& matrix, int column_index);
    void deleteRow(std::vector<std::vector<double>>& matrix, int row_index);
    std::vector<std::vector<double>> vector1DtoColumnVector(const std::vector<double>& vector);
    std::vector<double> columnVectortoVector1D(const std::vector<std::vector<double>>& col_vector);
    std::vector<double> scaleVector(const std::vector<double>& v1, const double& scalar);
    void addOnesToFront(std::vector<std::vector<double>>& matrix);
    std::vector<double> getColumn(const std::vector<std::vector<double>>& matrix, const int& col_index);
    std::vector<double> getRow(const std::vector<std::vector<double>>& matrix, const int& row_index);
    double accumulateVector(const std::vector<double>& v1);
    void updateColumn(std::vector<std::vector<double>>& matrix, std::vector<double> v1, int col_index);
    std::vector<double> createVector(const double& num, const int& length);
    std::vector<double> divideVector(const std::vector<double>& v1, const double& scalar);
    void addElement(std::vector<double>& v1, double value, int col_index);
    std::vector<std::vector<double>> hadamardProduct(const std::vector<std::vector<double>>& col_v1, const std::vector<std::vector<double>>& col_v2);
    std::vector<std::vector<double>> outerProduct(
        const std::vector<std::vector<double>>& col_vec,
        const std::vector<std::vector<double>>& row_vec);
    std::vector<std::vector<double>> subtractColumnVectors(const std::vector<std::vector<double>>& v1, const std::vector<std::vector<double>>& v2);
    double calculateMatrixEuclideanNorm(const std::vector<std::vector<double>> &matrix);
    std::vector<std::vector<double>> subtractMatrices(const std::vector<std::vector<double>> &m1, const std::vector<std::vector<double>> &m2);
    std::vector<std::vector<double>> createOnesMatrix(int num_rows, int num_cols);
 }


#endif