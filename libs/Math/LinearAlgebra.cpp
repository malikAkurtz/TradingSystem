#include <vector>
#include <cmath>
#include <stdexcept>


float innerProduct(std::vector<float> v1, std::vector<float> v2) {
    if (v1.size() != v2.size()) {
        throw std::invalid_argument("Matrix dimensions do not match");
    }
    
    int num_elements = v1.size();
    float innerProduct = 0;

    for (int i = 0; i < num_elements; i++) {
        innerProduct += (v1[i] * v2[i]);
    }

    return innerProduct;
}

std::vector<std::vector<float>> takeTranspose(std::vector<std::vector<float>> inputMatrix) {
    int numRows = inputMatrix.size();
    int numCols = inputMatrix[0].size();

    std::vector<std::vector<float>> transposed(numCols, std::vector<float>(numRows));

    for (int i = 0; i < numRows; i++) {
        for (int j = 0; j < numCols; j++) {
            transposed[j][i] = inputMatrix[i][j];
        }
    }

    return transposed;
}

std::vector<std::vector<float>> vectorToMatrix(std::vector<float> vector) {
    std::vector<std::vector<float>> asMatrix(vector.size(), std::vector<float>(1));

    for (int i = 0; i < vector.size(); i++) {
        asMatrix[i][0] = vector[i];
    }

    return asMatrix;
}

std::vector<float> matrixToVector(std::vector<std::vector<float>> matrix) {
    std::vector<float> asVector(matrix.size());
    for (int i = 0; i < matrix.size(); i++) {
        asVector[i] = matrix[i][0];
    }
    return asVector;
}


std::vector<float> addVectors(std::vector<float> v1, std::vector<float> v2) {
    if (v1.size() != v2.size()) {
        throw std::invalid_argument("Matrix dimensions do not match");
    }
    int num_elements = v1.size();

    std::vector<float> resultant(num_elements);

    for (int i = 0; i < num_elements; i++) {
        resultant[i] = v1[i]+v2[i];
    }

    return resultant;
}

std::vector<float> subtractVectors(std::vector<float> v1, std::vector<float> v2) {
    if (v1.size() != v2.size()) {
        throw std::invalid_argument("Vector dimensions do not match");
    }
    int num_elements = v1.size();

    std::vector<float> resultant(num_elements);

    for (int i = 0; i < num_elements; i++) {
        resultant[i] = v1[i]-v2[i];
    }

    return resultant;
}

float calculateNorm(std::vector<float> v1) {
    return sqrt(innerProduct(v1, v1));
}

std::vector<float> scaleVector(std::vector<float> v1, float scalar) {
    int num_elements = v1.size();

    std::vector<float> v1_scaled(num_elements);

    for (int i = 0; i < num_elements; i++) {
        v1_scaled[i] = v1[i] * scalar;
    }

    return v1_scaled;
}



std::vector<std::vector<float>> matrixMultiply(std::vector<std::vector<float>> m1, std::vector<std::vector<float>> m2) {
    // if m1 #cols != m2 #rows
    if (m1[0].size() != m2.size()) {
        throw std::invalid_argument("Matrix dimensions do not match");
    }

    std::vector<std::vector<float>> resultant(m1.size(), std::vector<float>(m2[0].size()));

    std::vector<std::vector<float>>  m2_T = takeTranspose(m2);
    // for every row in m1
    for (int i = 0; i < m1.size(); i++) {
        // for every column in m2 aka every row in the transpose
        for (int j = 0; j < m2_T.size(); j++) {
            resultant[i][j] = innerProduct(m1[i], m2_T[j]);
        }
    }

    return resultant;
}

std::vector<float> getRow(std::vector<std::vector<float>> matrix, int row_index) {
    return matrix[row_index];
}

std::vector<float> getColumn(std::vector<std::vector<float>> matrix, int col_index) {
    int num_rows = matrix.size();

    std::vector<float> column_vector(num_rows);

    for (int i = 0; i < num_rows; i++) {
        column_vector[i] = matrix[i][col_index];
    }

    return column_vector;
}

void addRow(std::vector<std::vector<float>>& matrix, std::vector<float> row) {
    matrix.push_back(row);
}

void addOnesToFront(std::vector<std::vector<float>>& matrix) {
    for (int i = 0; i < matrix.size(); i++) {
        matrix[i].insert(matrix[i].begin(), 1.0);
    }

}
void addColumn(std::vector<std::vector<float>>& matrix, std::vector<float> column) {
    for (int i = 0; i < matrix.size(); i++) {
        matrix[i].push_back(column[i]);
    }
}

void deleteColumn(std::vector<std::vector<float>>& matrix, int column_index) {
    for (int i = 0; i < matrix.size(); i++) {
        matrix[i].erase(matrix[i].begin() + column_index);
    }
}

void deleteRow(std::vector<std::vector<float>>& matrix, int row_index) {
    matrix.erase(matrix.begin() + row_index);
}

std::vector<float> solveSystem(std::vector<std::vector<float>> matrix, std::vector<float> b) {
    if (matrix.size() != matrix[0].size()) {
        throw std::invalid_argument("Matrix dimensions do not match");
    }
    int solution_size = b.size();
    int num_rows = matrix.size();
    int num_cols = num_rows;

    std::vector<std::vector<float>> gaussian = matrix;
    addColumn(gaussian, b);

    std::vector<float> solution(solution_size, 0);

    for (int i = 0; i < num_rows; i++) {
        for (int j = 0; j < num_cols; j++) {
            //if we are at a diagonal entry, divide it by itself to make it 1
            if (i == j) {
                std::vector<float> original_row = gaussian[i];
                if (original_row[j] == 0) {
                    throw std::invalid_argument("Can't have a zero on diagonal of matrix");
                }
                gaussian[i] = scaleVector(original_row, (1 / original_row[j]));

                for (int row = i+1; row < num_rows; row++) {
                    std::vector<float> this_row = gaussian[row];
                    std::vector<float> new_row = subtractVectors(this_row, scaleVector(gaussian[i], this_row[j]));

                    gaussian[row] = new_row;
                }
            }
        }
    }

    std::vector<std::vector<float>> gaussian_T = takeTranspose(gaussian);


    std::vector<float> constants = gaussian_T[gaussian_T.size()-1];


    deleteColumn(gaussian, gaussian[0].size()-1);

    // population first solution vector value
    solution[solution_size-1] = constants[solution_size-1];


    for (int i = solution_size-2; i >= 0; i--) {
        solution[i] = constants[i] - innerProduct(gaussian[i], solution);
    }

    return solution;
}

