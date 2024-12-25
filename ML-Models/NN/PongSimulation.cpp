
#include "PongSimulation.h"


PongSimulator::PongSimulator(const NeuralNetwork& NN_brain) 
{
    reset();
    paddle_NN_brain = NN_brain;
}

void PongSimulator::reset()
{
    ball_pos.first = WINDOW_WIDTH / 2;
    ball_pos.second = WINDOW_HEIGHT / 2;

    ball_vel.first = 1.0;
    ball_vel.second = 1.0;

    paddle_NN.first = WINDOW_WIDTH * 0.9;
    paddle_NN.second = WINDOW_HEIGHT / 2;

    paddle_opp.first = WINDOW_WIDTH * 0.1;
    paddle_opp.second = WINDOW_HEIGHT / 2;
    num_steps = 1000;
    NN_score = 0;
    opp_score = 0;
    NN_fitness = 0;
    done = false;
}

void PongSimulator::play()
{   
    if (!done)
    {
        // if prediction greater than 0.5, move down, if prediction < 0.5, move up
        double decision = paddle_NN_brain.getPredictions({getGameState()})[0][0];
        if (decision >= 0.5) // decision >= 0.5 means increase y i.e move down
        {
            decision = 1;
        }
        else
        {
            decision = -1;
        }

        // get decision for NN paddle
        paddle_NN.second += PADDLE_SPEED * decision;
        // get decision for opponent paddle
        if (paddle_opp.second < ball_pos.second) 
        {
            paddle_opp.second += PADDLE_SPEED;
        } 
        else
        {
            paddle_opp.second -= PADDLE_SPEED;
        }


        // move ball
        ball_pos.first += BALL_SPEED * ball_vel.first;
        ball_pos.second += BALL_SPEED * ball_vel.second;

        if (paddle_NN.second <= 0)
        {
            paddle_NN.second = 0;
        } else if (paddle_NN.second >= 100)
        {
            paddle_NN.second = 100;
        }

        // ball collision with right (Neural Network) paddle
        if ((std::fabs(ball_pos.first - paddle_NN.first) < 1.5) // if ball x position matches NN paddle x position
        && ((paddle_NN.second - PADDLE_HEIGHT) <= ball_pos.second) // and NN paddle y - paddle height <= ball y
        && (ball_pos.second <= (paddle_NN.second + PADDLE_HEIGHT))) // and ball y <= NN paddle y + paddle height
        {
            ball_vel.first *= -1; // flip x velocity
            NN_fitness += 5;
        }
        // ball collision with left (opponent) paddle
        else if ((std::fabs(ball_pos.first - paddle_opp.first) < 1.5) // if ball x position matches opp paddle x position
        && (paddle_opp.second - PADDLE_HEIGHT <= ball_pos.second) // and opp paddle y - paddle height <= ball y
        && (ball_pos.second <= paddle_opp.second + PADDLE_HEIGHT)) // and ball y <= opp paddle y + paddle height
        {   
            //
            ball_vel.first *= -1; // flip x velocity
        }
        // ball made it past NN paddle
        else if (ball_pos.first > paddle_NN.first)
        {
            opp_score += 1;
            NN_fitness -= 10;
            // reset the ball
            ball_pos.first = WINDOW_WIDTH / 2;
            ball_pos.second = WINDOW_HEIGHT / 2;

            ball_vel.first = 1.0;
            ball_vel.second = 1.0;
        }
        else if (ball_pos.first < paddle_opp.first)
        {
            NN_score += 1;
            NN_fitness += 10;

            ball_pos.first = WINDOW_WIDTH / 2;
            ball_pos.second = WINDOW_HEIGHT / 2;

            ball_vel.first = 1.0;
            ball_vel.second = 1.0;
        }

        if (ball_pos.second >= 100 || ball_pos.second <= 0)
        {
            ball_vel.second *= -1;
        }
        NN_fitness++;
        num_steps -= 1;
        if (num_steps == 0)
        {
            done = true;
        }
    }
    
}

std::vector<double> PongSimulator::getGameState()
{
    std::vector<double> gameState(8);
    gameState[0] = ball_pos.first;
    gameState[1] = ball_pos.second;
    gameState[2] = ball_vel.first;
    gameState[3] = ball_vel.second;
    gameState[4] = paddle_NN.first;
    gameState[5] = paddle_NN.second;
    gameState[6] = paddle_opp.first;
    gameState[7] = paddle_opp.second;
    return gameState;
}

int PongSimulator::getFitness()
{
    return NN_fitness;
}