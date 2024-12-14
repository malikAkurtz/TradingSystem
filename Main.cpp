#include "ML-Models/linear_regression.cpp"
#include <iostream>
#include "LinearAlgebra.h"
#include "Output.h"


// int main() {
//     std::vector<std::vector<float>> features_matrix = {{1, 2, 3, 4, 5, 6, 7, 8, 9, 10}, 
//                                                         {2, 3, 4, 5, 6, 7, 8, 9, 10, 11}};
//     std::vector<float> labels = {12.0f, 17.0f, 22.0f, 27.0f, 32.0f, 37.0f, 42.0f, 47.0f, 52.0f, 57.0f};
//     LinearRegression LR(0.0001);
//     LR.fit(features_matrix, labels);
    
//     std::vector<float> predictions = LR.getPredictions(features_matrix);;
//     for (int i = 0; i < predictions.size(); i++) {
//         std::cout << "Prediction" << predictions[i] << "Actual" << labels[i]<< std::endl;
//     }
//     return 0;
// }

int main() {
    std::vector<std::vector<float>> m1 = {
    {1, 2},
    {3, 4}
    };

    std::vector<std::vector<float>> m2 = {
    {5, 6},
    {7, 8}
    };

    std::vector<std::vector<float>> resultant = matrixMultiply(m1, m2);

    printMatrix(resultant);


}