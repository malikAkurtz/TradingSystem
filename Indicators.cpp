#include <vector>
#include <stdexcept>


double calculateCurSMA(std::vector<double> prices, int n) {
    double cumSum = 0;
    for (int i = 0; i < n; i++) {
        cumSum += prices[i];
    }
    return cumSum / n;
}

std::vector<double> calculateSMAValues(std::vector<double> prices, int n) {
    if (n > prices.size()) {
        throw std::invalid_argument("period can't be greater than length of price vector");
    }
    else {
        std::vector<double> SMA_Values;

        for (int i = 0; (i + n) <= prices.size(); i++) {

            std::vector<double>::const_iterator first = prices.begin() + i;
            std::vector<double>::const_iterator last = prices.begin() + (n + i);
            std::vector<double> n_elements(first, last);
            SMA_Values.push_back(calculateCurSMA(n_elements, n));
        }
        return SMA_Values;
    }

}

double calculateCurEMA(double current_price, double Prev_EMA_val, double multiplier) {
    return (current_price * multiplier) + (Prev_EMA_val * (1-multiplier));
}

std::vector<double> calculateEMAValues(std::vector<double> prices, int n, int smoothing = 2) {
    if (n > prices.size()) {
        throw std::invalid_argument("period can't be greater than length of price vector");
    }
    else {

        std::vector<double> EMA_Values;

        double multiplier = (double) smoothing / (n + 1);

        std::vector<double>::const_iterator first = prices.begin();
        std::vector<double>::const_iterator last = prices.begin() + n;
        std::vector<double> first_n_vector(first, last);
        double EMA_Start = calculateCurSMA(first_n_vector, n);

        EMA_Values.push_back(EMA_Start);

        for (int i = n; i < prices.size(); i++) {
            double curPrice = prices[i];
            double prevEMA = EMA_Values[i-n];
            double curEMA = calculateCurEMA(curPrice, prevEMA, multiplier);
            EMA_Values.push_back(curEMA);
        }

        return EMA_Values;

    }
}


