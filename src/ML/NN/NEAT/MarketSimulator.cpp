
#include "MarketSimulator.h"
#include "Entity.h"

MarketSimulator::MarketSimulator(Entity &entity, const std::vector<std::vector<double>> &features_matrix) : entity(&entity), features_matrix(features_matrix) {};

void MarketSimulator::setup()
{
    this->cash = STARTING_CASH;
    this->units = 0;
    this->portfolio_value = STARTING_CASH;
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

    for (int i = 1; i < features_matrix.size(); i++)
    {
        debugMessage("--------------------New Day--------------------", "");
        
        double decision = makeDecision({features_matrix[i - 1]})[0][0];
        debugMessage("Decision is: " + std::to_string(decision), "");

        double open_price = features_matrix[i][1];
        debugMessage("Open Price is: " + std::to_string(open_price), "");
        //double close_price = features_matrix[i][0];
        // if the decision is below -0.2, sell 
        if (decision < -0.2)
        {   

            int units_to_sell = floor(CAPITAL_AT_RISK / open_price);

            if (this->units >= units_to_sell)
            {
                debugMessage("Selling " + std::to_string(units_to_sell) + " Units @ " + std::to_string(open_price), "");

                this->executeSellOrder(units_to_sell, open_price, SLIPPAGE);

                debugMessage("Cash on Hand: " + std::to_string(this->cash), "");
                debugMessage("Units Held: " + std::to_string(this->units), "");
            }
            else
            {
                units_to_sell = this->units;

                debugMessage("Selling " + std::to_string(units_to_sell) + " Units @ " + std::to_string(open_price), "");

                this->executeSellOrder(units_to_sell, open_price, SLIPPAGE);

                debugMessage("Cash on Hand: " + std::to_string(this->cash), "");
                debugMessage("Units Held: " + std::to_string(this->units), "");
            }
        }
        else if (decision > 0.2)
        {

            int units_to_buy = floor(CAPITAL_AT_RISK / open_price);
            double trade_cost = units_to_buy * (open_price * (1 + SLIPPAGE));

            if (this->cash >= trade_cost)
            {   
                debugMessage("Buying " + std::to_string(units_to_buy) + " Units @ " + std::to_string(open_price), "");

                this->executeBuyOrder(units_to_buy, open_price, SLIPPAGE);

                debugMessage("Cash on Hand: " + std::to_string(this->cash), "");
                debugMessage("Units Held: " + std::to_string(this->units), "");
            }
            else
            {
                units_to_buy = floor(this->cash / open_price);

                debugMessage("Buying " + std::to_string(units_to_buy) + " Units @ " + std::to_string(open_price), "");

                this->executeBuyOrder(units_to_buy, open_price, SLIPPAGE);

                debugMessage("Cash on Hand: " + std::to_string(this->cash), "");
                debugMessage("Units Held: " + std::to_string(this->units), "");
            }
        }
        else 
        {
        }
        debugMessage("----------------End of Day----------------", "");
    }

    this->portfolio_value = this->calculatePortfolioValue(this->cash, this->units, features_matrix.back()[1]);
    debugMessage("Final Portfolio Value: " + std::to_string(this->portfolio_value), "");
    double percentage_gain = (this->portfolio_value - STARTING_CASH) / STARTING_CASH;
    debugMessage("Overall Percentage Gain: " + std::to_string(percentage_gain), "");
    return this->portfolio_value;
}

void MarketSimulator::executeBuyOrder(int units_to_buy, double price_per_unit, float slippage)
{
    double trade_cost = units_to_buy * (price_per_unit * (1 + slippage));
    this->cash -= trade_cost;
    this->units += units_to_buy;
}

void MarketSimulator::executeSellOrder(int units_to_sell, double price_per_unit, float slippage)
{
    this->cash += units_to_sell * (price_per_unit * (1 - slippage));
    this->units -= units_to_sell;
}

double MarketSimulator::calculatePortfolioValue(int cash_on_hand, int units_held, double last_price_per_unit)
{
    return cash_on_hand + (units_held * last_price_per_unit);
}
