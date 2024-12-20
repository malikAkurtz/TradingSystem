#ifndef OUTPUT
#define OUTPUT

#include <vector>

extern bool DEBUG;

void printVector(const std::vector<double>& vec);
void printMatrix(const std::vector<std::vector<double>>& matrix);
void print(std::string toPrint);
void printPredictionsVSLabels(const std::vector<std::vector<std::vector<double>>>& predictions, 
                              const std::vector<std::vector<double>>& labels);
void printMatrixShape(std::vector<std::vector<double>> matrix);
void printVectorShape(std::vector<double> vector);
void printMatrix_(const std::vector<std::vector<double>>& matrix);

#endif