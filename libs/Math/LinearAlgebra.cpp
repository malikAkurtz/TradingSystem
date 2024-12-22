#include <vector>
#include <cmath>
#include <stdexcept>
#include <Output.h>


double innerProduct(const std::vector<double>& v1, const std::vector<double>& v2) {
    if (v1.size() != v2.size()) {
        throw std::invalid_argument("Matrix dimensions do not match on call to innerProduct()");
    }
    
    int num_elements = v1.size();
    double innerProduct = 0;

    for (int i = 0; i < num_elements; i++) {
        innerProduct += (v1[i] * v2[i]);
    }

    return innerProduct;
}

std::vector<std::vector<double>> takeTranspose(const std::vector<std::vector<double>>& inputMatrix) {
    int numRows = inputMatrix.size();
    int numCols = inputMatrix[0].size();

    std::vector<std::vector<double>> transposed(numCols, std::vector<double>(numRows));

    for (int i = 0; i < numRows; i++) {
        for (int j = 0; j < numCols; j++) {
            transposed[j][i] = inputMatrix[i][j];
        }
    }

    return transposed;
}

std::vector<std::vector<double>> vector1DtoColumnVector(const std::vector<double>& vector) {
    std::vector<std::vector<double>> col_vector(vector.size(), std::vector<double>(1));

    for (int i = 0; i < vector.size(); i++) {
        col_vector[i][0] = vector[i];
    }

    return col_vector;
}

std::vector<double> columnVectortoVector1D(const std::vector<std::vector<double>>& col_vector) {
    std::vector<double> vector1D(col_vector.size());
    for (int i = 0; i < col_vector.size(); i++) {
        vector1D[i] = col_vector[i][0];
    }
    return vector1D;
}


std::vector<double> addVectors(const std::vector<double>& v1, const std::vector<double>& v2) {
    if (v1.size() != v2.size()) {
        throw std::invalid_argument("Vector dimensions do not match on call to addVectors()");
    }
    int num_elements = v1.size();

    std::vector<double> resultant(num_elements);

    for (int i = 0; i < num_elements; i++) {
        resultant[i] = v1[i]+v2[i];
    }

    return resultant;
}

std::vector<double> subtractVectors(const std::vector<double>& v1, const std::vector<double>& v2) {
    if (v1.size() != v2.size()) {
        throw std::invalid_argument("Vector dimensions do not match");
    }
    int num_elements = v1.size();

    std::vector<double> resultant(num_elements);

    for (int i = 0; i < num_elements; i++) {
        resultant[i] = v1[i]-v2[i];
    }

    return resultant;
}

double calculateNorm(const std::vector<double>& v1) {
    return sqrt(innerProduct(v1, v1));
}

std::vector<double> scaleVector(const std::vector<double>& v1, const double& scalar) {
    int num_elements = v1.size();

    std::vector<double> v1_scaled(num_elements);

    for (int i = 0; i < num_elements; i++) {
        v1_scaled[i] = v1[i] * scalar;
    }

    return v1_scaled;
}

std::vector<double> divideVector(const std::vector<double>& v1, const double& scalar) {
    int num_elements = v1.size();

    std::vector<double> v1_scaled(num_elements);

    for (int i = 0; i < num_elements; i++) {
        v1_scaled[i] = v1[i] / scalar;
    }

    return v1_scaled;
}



std::vector<std::vector<double>> matrixMultiply(const std::vector<std::vector<double>>& m1, const std::vector<std::vector<double>>& m2) {
    // if m1 #cols != m2 #rows
    if (m1[0].size() != m2.size()) {
        throw std::invalid_argument("Matrix dimensions do not match on call to matrixMultiply()");
    }

    int rows = m1.size();
    int cols = m2[0].size();
    int inner_dimension = m2.size();
    std::vector<std::vector<double>> resultant(rows, std::vector<double>(cols));
    

    for (int i = 0; i < m1.size(); i++) {
        for (int j = 0; j < m2[0].size(); j++) {
            for (int k = 0; k < inner_dimension; k++) {
                resultant[i][j] += m1[i][k] * m2[k][j];
            }
        }
    }

    return resultant;
}

std::vector<double> getRow(const std::vector<std::vector<double>>& matrix, const int& row_index) {
    return matrix[row_index];
}

std::vector<double> getColumn(const std::vector<std::vector<double>>& matrix, const int& col_index) {
    int num_rows = matrix.size();

    std::vector<double> column_vector(num_rows);

    for (int i = 0; i < num_rows; i++) {
        column_vector[i] = matrix[i][col_index];
    }

    return column_vector;
}

void addRow(std::vector<std::vector<double>>& matrix, std::vector<double> row) {
    matrix.push_back(row);
}

void addOnesToFront(std::vector<std::vector<double>>& matrix) {
    for (int i = 0; i < matrix.size(); i++) {
        matrix[i].insert(matrix[i].begin(), 1.0);
    }

}
void addColumn(std::vector<std::vector<double>>& matrix, std::vector<double> column) {
    for (int i = 0; i < matrix.size(); i++) {
        matrix[i].push_back(column[i]);
    }
}

void deleteColumn(std::vector<std::vector<double>>& matrix, int column_index) {
    for (int i = 0; i < matrix.size(); i++) {
        matrix[i].erase(matrix[i].begin() + column_index);
    }
}

void deleteRow(std::vector<std::vector<double>>& matrix, int row_index) {
    matrix.erase(matrix.begin() + row_index);
}

std::vector<double> solveSystem(std::vector<std::vector<double>> matrix, std::vector<double> b) {
    if (matrix.size() != matrix[0].size()) {
        throw std::invalid_argument("Matrix dimensions do not match on call to solveSystem()");
    }
    int solution_size = b.size();
    int num_rows = matrix.size();
    int num_cols = num_rows;

    std::vector<std::vector<double>> gaussian = matrix;
    addColumn(gaussian, b);

    std::vector<double> solution(solution_size, 0);

    for (int i = 0; i < num_rows; i++) {
        for (int j = 0; j < num_cols; j++) {
            //if we are at a diagonal entry, divide it by itself to make it 1
            if (i == j) {
                std::vector<double> original_row = gaussian[i];
                if (original_row[j] == 0) {
                    throw std::invalid_argument("Can't have a zero on diagonal of matrix");
                }
                gaussian[i] = scaleVector(original_row, (1 / original_row[j]));

                for (int row = i+1; row < num_rows; row++) {
                    std::vector<double> this_row = gaussian[row];
                    std::vector<double> new_row = subtractVectors(this_row, scaleVector(gaussian[i], this_row[j]));

                    gaussian[row] = new_row;
                }
            }
        }
    }

    std::vector<std::vector<double>> gaussian_T = takeTranspose(gaussian);


    std::vector<double> constants = gaussian_T[gaussian_T.size()-1];


    deleteColumn(gaussian, gaussian[0].size()-1);

    // population first solution vector value
    solution[solution_size-1] = constants[solution_size-1];


    for (int i = solution_size-2; i >= 0; i--) {
        solution[i] = constants[i] - innerProduct(gaussian[i], solution);
    }

    return solution;
}


double accumulateVector(const std::vector<double>& v1) {
    double cumSum = 0.0;
    for (int i = 0; i < v1.size(); i++) {
        cumSum += v1[i];
    }
    return cumSum;
}

void updateColumn(std::vector<std::vector<double>>& matrix, std::vector<double> v1, int col_index) {

    //for every row
    for (int i = 0; i < matrix.size(); i++) {
        //update the col index
        matrix[i][col_index] = v1[i];
    }
}

std::vector<double> createVector(const double& num, const int& length) {
    std::vector<double> v1(length, num);
    return v1;
}

std::vector<std::vector<double>> createColumnVector(const double& num, const int& length) {
    std::vector<std::vector<double>> col_vec(length, std::vector<double>(1));
    for (int i = 0; i < length; i++) {
        col_vec[i][0] = num;
    }

    return col_vec;
}

void addElement(std::vector<double>& v1, double value, int col_index) {
    v1.insert(v1.begin() + col_index, value);
}



// takes two matrices, returns a matrix
std::vector<std::vector<double>> hadamardProduct(const std::vector<std::vector<double>>& m1, const std::vector<std::vector<double>>& m2) {
    if (m1.size() != m2.size() || m1[0].size() != m2[0].size()) {
        throw std::invalid_argument("Matrix dimensions do not match for hadamardProduct.");
    }

    int num_rows = m1.size();
    int num_cols = m1[0].size();

    std::vector<std::vector<double>> resultant(num_rows, std::vector<double>(num_cols));

    for (int i = 0; i < num_rows; i++) {
        for (int j = 0; j < num_cols; j++) {
            resultant[i][j] = m1[i][j] * m2[i][j];
        }
        
    }

    return resultant;

}

std::vector<std::vector<double>> subtractColumnVectors(const std::vector<std::vector<double>>& v1, const std::vector<std::vector<double>>& v2) {
    if (v1.size() != v2.size()) {
        throw std::invalid_argument("Vector dimensions do not match for subtraction.");
    }
    std::vector<std::vector<double>> resultant(v1.size(), std::vector<double>(1));

    for (int i = 0; i < resultant.size(); i++) {
        resultant[i][0] = v1[i][0] - v2[i][0];
    }

    return resultant;
}

std::vector<std::vector<double>> outerProduct(
    const std::vector<std::vector<double>>& col_vec,
    const std::vector<std::vector<double>>& row_vec) {
    // Validate input: col_vec must be (n, 1), row_vec must be (1, m)
    if (col_vec[0].size() != 1 || row_vec.size() != 1) {
        throw std::invalid_argument("Invalid dimensions for outer product");
    }

    int n = col_vec.size();     // Rows in col_vec
    int m = row_vec[0].size();  // Columns in row_vec
    std::vector<std::vector<double>> result(n, std::vector<double>(m, 0.0));

    // Compute outer product
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < m; j++) {
            result[i][j] = col_vec[i][0] * row_vec[0][j];
        }
    }

    return result;
}

double calculateMatrixEuclideanNorm(const std::vector<std::vector<double>>& matrix) {
    double cumSum = 0;
    for (int i = 0; i < matrix.size(); i++) {
        for (int j = 0; j < matrix[0].size(); j++) {
            cumSum += (matrix[i][j] * matrix[i][j]);
        }
    }
    return sqrt(cumSum);
}

std::vector<std::vector<double>> subtractMatrices(const std::vector<std::vector<double>>& m1, const std::vector<std::vector<double>>& m2) {
    if (m1.size() != m2.size() || m1[0].size() != m2[0].size()) {
        throw std::invalid_argument("Matrix dimensions do not match for subtraction.");
    }

    int num_rows = m1.size();
    int num_cols = m1[0].size();
    std::vector<std::vector<double>> resultantMatrix(num_rows, std::vector<double>(num_cols));

    for (int i = 0; i < num_rows; i++) {
        for (int j = 0; j < num_cols; j++) {
            resultantMatrix[i][j] = m1[i][j] - m2[i][j];
        }
    }

    return resultantMatrix;
}