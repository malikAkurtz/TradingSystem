
#include "MarketSimulator.h"
#include "Entity.h"

MarketSimulator::MarketSimulator(Entity &entity, const std::vector<std::vector<double>> &features_matrix, const std::vector<std::vector<double>> &labels) : entity(&entity), features_matrix(features_matrix), labels(labels) {};

void MarketSimulator::setup()
{
    this->cash = 10000;
    this->units = 0;
}

std::vector<std::vector<double>> MarketSimulator::makeDecision(const std::vector<std::vector<double>> &inputs)
{
    return entity->brain.feedForward(inputs);
}

double MarketSimulator::simulate()
{
    // for every row (day) in xrp price data
    // use the data from yesterday to place a trade at market open today, hence why starting at i = 1
    this->setup();
    
    double decision;
    // decision will be as follows:
    // if output is < -0.5, sell (1 * output)
    // if output is > 0.5, buy (1 * output)
    for (int i = 1; i < features_matrix.size(); i++)
    {
        decision = makeDecision({features_matrix[i - 1]})[0][0];
        if (decision < -1)
        {
            units += (1 * decision);
            cash += units * (features_matrix[i][1]);
        }
        else if (decision > 1)
        {
            units += (1 * decision);
            cash -= units * (features_matrix[i][1]);
        }
        else 
        {
        }
    }
    return cash;
}