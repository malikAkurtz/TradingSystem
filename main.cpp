#include "Indicators.h"
#include <iostream>

int main() {
    std::vector<double> numbers = {20, 22, 24, 25, 23, 26, 28, 26, 29, 27, 28, 30, 27, 29, 28};

    std::cout << calculateSMA(numbers, 15);
}