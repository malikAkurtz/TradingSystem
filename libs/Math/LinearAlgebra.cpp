#include <vector>
#include <cmath>

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
        resultant[i] = v2[i]-v1[i];
    }

    return resultant;
}

float calculateNorm(std::vector<float> v1) {
    int num_elements = v1.size();
    float sumOfSquares = 0;

    for (int i = 0; i < num_elements; i++) {
        sumOfSquares += pow(v1[i], 2);
    }

    return sqrt(sumOfSquares);
}