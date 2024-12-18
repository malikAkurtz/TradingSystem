#ifndef OUTPUT
#define OUTPUT

#include <vector>

void printVector(const std::vector<double>& vec);
void printMatrix(const std::vector<std::vector<double>>& matrix);
void print(std::string toPrint);
void printPredictionsVSLabels(std::vector<double> predictions, std::vector<double> labels);

#endif