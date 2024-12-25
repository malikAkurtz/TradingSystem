#ifndef SIMULATOR_H
#define SIMULATOR_H

#include <vector>
#include "NeuralNetwork.h"

static const int WINDOW_WIDTH  = 100;
static const int WINDOW_HEIGHT = 100;
static const int PADDLE_HEIGHT = 20;
static const int BALL_SPEED    = 2;
static const int PADDLE_SPEED  = 5;

class PongSimulator
{
public:
    NeuralNetwork paddle_NN_brain;

    std::pair<double, double> ball_pos;
    std::pair<double, double> ball_vel;

    std::pair<double, double> paddle_NN;
    std::pair<double, double> paddle_opp;

    int num_steps;
    int NN_score;
    int opp_score;

    int NN_fitness;

    bool done;

    PongSimulator(const NeuralNetwork &NN_brain);

    void reset();

    void play();

    std::vector<double> getGameState();

    int getFitness();
};

#endif