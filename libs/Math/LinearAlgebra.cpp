#include <vector>
#include <cmath>
#include <stdexcept>


float innerProduct(std::vector<float> v1, std::vector<float> v2) {
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


std::vector<float> addVectors(std::vector<float> v1, std::vector<float> v2) {
    int num_elements = v1.size();

    std::vector<float> resultant(num_elements);

    for (int i = 0; i < num_elements; i++) {
        resultant[i] = v1[i]+v2[i];
    }

    return resultant;
}

std::vector<float> subtractVectors(std::vector<float> v1, std::vector<float> v2) {
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



std::vector<std::vector<float>> matrixMultiply(std::vector<std::vector<float>> m1, std::vector<std::vector<float>> m2) {
    // if m1 #cols != m2 #rows
    if (m1[0].size() != m2.size()) {
        throw std::invalid_argument("Matrix dimensions do not match");
    }

    std::vector<std::vector<float>> resultant(m1.size(), std::vector<float>(m2[0].size()));

    std::vector<std::vector<float>>  m2_T = takeTranspose(m2);
    // for every row in m1
    for (int i = 0; i < m1.size(); i++) {
        // for every column in m2
        for (int j = 0; j < m2[0].size(); j++) {
            resultant[i][j] = innerProduct(m1[i], m2_T[j]);
        }
    }

    return resultant;
}