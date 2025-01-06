#ifndef MARKET_SIM_H
#define MARKET_SIM_H

#include <vector>
#include <cmath>

struct Entity;

const int STARTING_CASH = 2000;
const float SLIPPAGE = 0.005;
const double SELL_THRESHOLD = -0.2;
const double BUY_THRESHOLD = 0.2;
const int CAPITAL_AT_RISK = 500;

class MarketSimulator
{
public:
    double cash;
    int units;
    double portfolio_value;

    Entity* entity;
    std::vector<std::vector<double>> features_matrix;

    MarketSimulator(Entity &entity, const std::vector<std::vector<double>> &features_matrix);

    void setup();

    std::vector<std::vector<double>> makeDecision(const std::vector<std::vector<double>> &inputs);

    double simulate();

    void executeBuyOrder(int units_to_buy, double price_per_unit, float slippage);

    void executeSellOrder(int units_to_buy, double price_per_unit, float slippage);

    double calculatePortfolioValue(int cash_on_hand, int units_held, double last_price_per_unit);
};

#endif