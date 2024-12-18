#include <vector>
#include <cmath>
#include <stdexcept>


double innerProduct(std::vector<double> v1, std::vector<double> v2) {
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

std::vector<std::vector<double>> takeTranspose(std::vector<std::vector<double>> inputMatrix) {
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

std::vector<std::vector<double>> vectorToMatrix(std::vector<double> vector) {
    std::vector<std::vector<double>> asMatrix(vector.size(), std::vector<double>(1));

    for (int i = 0; i < vector.size(); i++) {
        asMatrix[i][0] = vector[i];
    }

    return asMatrix;
}

std::vector<double> matrixToVector(std::vector<std::vector<double>> matrix) {
    std::vector<double> asVector(matrix.size());
    for (int i = 0; i < matrix.size(); i++) {
        asVector[i] = matrix[i][0];
    }
    return asVector;
}


std::vector<double> addVectors(std::vector<double> v1, std::vector<double> v2) {
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

std::vector<double> subtractVectors(std::vector<double> v1, std::vector<double> v2) {
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

double calculateNorm(std::vector<double> v1) {
    return sqrt(innerProduct(v1, v1));
}

std::vector<double> scaleVector(std::vector<double> v1, double scalar) {
    int num_elements = v1.size();

    std::vector<double> v1_scaled(num_elements);

    for (int i = 0; i < num_elements; i++) {
        v1_scaled[i] = v1[i] * scalar;
    }

    return v1_scaled;
}

std::vector<double> divideVector(std::vector<double> v1, double scalar) {
    int num_elements = v1.size();

    std::vector<double> v1_scaled(num_elements);

    for (int i = 0; i < num_elements; i++) {
        v1_scaled[i] = v1[i] / scalar;
    }

    return v1_scaled;
}



std::vector<std::vector<double>> matrixMultiply(std::vector<std::vector<double>> m1, std::vector<std::vector<double>> m2) {
    // if m1 #cols != m2 #rows
    if (m1[0].size() != m2.size()) {
        throw std::invalid_argument("Matrix dimensions do not match on call to matrixMultiply()");
    }

    std::vector<std::vector<double>> resultant(m1.size(), std::vector<double>(m2[0].size()));

    std::vector<std::vector<double>>  m2_T = takeTranspose(m2);
    // for every row in m1
    for (int i = 0; i < m1.size(); i++) {
        // for every column in m2 aka every row in the transpose
        for (int j = 0; j < m2_T.size(); j++) {
            resultant[i][j] = innerProduct(m1[i], m2_T[j]);
        }
    }

    return resultant;
}

std::vector<double> getRow(std::vector<std::vector<double>> matrix, int row_index) {
    return matrix[row_index];
}

std::vector<double> getColumn(std::vector<std::vector<double>> matrix, int col_index) {
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


double accumulateVector(std::vector<double> v1) {
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

std::vector<double> createVector(double num, int length) {
    std::vector<double> v1(length, num);
    return v1;
}

void addElement(std::vector<double>& v1, double value, int col_index) {
    v1.insert(v1.begin() + col_index, value);
}


std::vector<double> hadamardProduct(std::vector<double> v1, std::vector<double> v2) {
    int num_elements = v1.size();
    if (num_elements != v2.size()) {
        throw std::invalid_argument("Vector dimensions do not match on call to hadamardProduct()");
    }

    std::vector<double> resultant(num_elements);

    for (int i = 0; i < num_elements; i++) {
        resultant[i] = v1[i] * v2[i];
    }

    return resultant;

}