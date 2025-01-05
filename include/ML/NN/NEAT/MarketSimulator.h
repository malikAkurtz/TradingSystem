#ifndef MARKET_SIM_H
#define MARKET_SIM_H

#include <vector>

struct Entity;

class MarketSimulator
{
public:
    double cash;
    int units;

    Entity* entity;
    std::vector<std::vector<double>> features_matrix;
    std::vector<std::vector<double>> labels;

    MarketSimulator(Entity &entity, const std::vector<std::vector<double>> &features_matrix, const std::vector<std::vector<double>> &labels);

    std::vector<std::vector<double>> makeDecision(const std::vector<std::vector<double>> &inputs);

    double simulate();

};

#endif