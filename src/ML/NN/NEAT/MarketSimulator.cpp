
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
        std::cout << "----------------New Market Open----------------" << std::endl;
        decision = makeDecision({features_matrix[i - 1]})[0][0];
        if (decision < -0.01)
        {
            // std::cout << "Selling" << std::endl;
            // double units_to_sell = abs((10 * decision));
            // this->units -= units_to_sell;
            // this->cash += units_to_sell * (features_matrix[i][1]);
        }
        else if (decision > 0.01)
        {
            double xrp_open_price = features_matrix[i][1];
            std::cout << "XRP Open Price: " << xrp_open_price << std::endl;
            double total_cash_available = this->cash;
            std::cout << "Total Cash Available: " << total_cash_available << std::endl;
            double cash_trade_size = decision * total_cash_available;
            std::cout << "Cash Trade Size: " << cash_trade_size << std::endl;
            int units_to_buy = floor(cash_trade_size / xrp_open_price);
            std::cout << "Units To Purcahse: " << units_to_buy << std::endl;
            this->units += units_to_buy;
            this->cash -= units_to_buy * (features_matrix[i][1]);
            std::cout << "Buying: " << units_to_buy << " XRP at Open Price of: " << features_matrix[i][1] << std::endl;
            std::cout << "Cash Balance is Now: " << this->cash << std::endl;

            // at market close, sell
            this->cash += this->units * (features_matrix[i][0]);
            this->units -= this->units;
            std::cout << "Selling XRP at Close Price of: " << features_matrix[i][0] << std::endl;
            std::cout << "Cash Balance is Now: " << this->cash << std::endl;
        }
        else 
        {
        }
        std::cout << "----------------Market Close----------------" << std::endl;
    }
    std::cout << "Final Units Held: " << units << std::endl;
    std::cout << "Final Cash Held: " << cash << std::endl;
    double portfolio_value = cash + (units * features_matrix.back()[1]);
    return portfolio_value;
}