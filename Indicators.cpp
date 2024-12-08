#include <vector>
#include <stdexcept>

double calculateSMA(std::vector<double> prices, int period) {
    if (period > prices.size()) {
        throw std::invalid_argument("period can't be greater than length of price vector");
    }
    else {
        int cumSum = 0;
        for (int i = 0; i < prices.size(); i++) {
            cumSum += prices[i];
        }
        return (double) cumSum / period;
    }
}

double calculateEMA(std::vector<double> prices, int period, int smoothing) {
    if (period > prices.size()) {
        throw std::invalid_argument("period can't be greater than length of price vector");
    }
    else {
        int latest_price = prices[prices.size() - 1];
        double multiplier = (double) smoothing / (prices.size() + 1);


    }
}
