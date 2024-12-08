#ifndef INDICATORS_H
#define INDICATORS_H

#include <vector>

double calculateSMA(std::vector<double> prices, int period);

double calculateEMA(std::vector<double> prices, int period);

#endif