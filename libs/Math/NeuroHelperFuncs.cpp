#include "NeuroHelperFuncs.h"

std::vector<double> uniformCrossover(const NeuralNetwork& p1, const NeuralNetwork& p2)
{
    std::vector<double> p1_encoding = p1.getNetworkEncoding();
    std::vector<double> p2_encoding = p2.getNetworkEncoding();

    int sequence_length = p1_encoding.size();

    std::vector<double> childEncoding(sequence_length);

    for (int i = 0; i < sequence_length; i++)
    {
        if (std::rand() % 2)
        {
            childEncoding[i] = p1_encoding[i];
        }
        else {
            childEncoding[i] = p2_encoding[i];
        }
    }

    return childEncoding;
}

void mutate(std::vector<double>& networkEncoding, const float& mutation_rate, const float& mutation_range)
{
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<> prob_dist(0.0, 1.0);
    std::uniform_real_distribution<> mutation_dist(-mutation_range, mutation_range);

    for (int i = 0; i < networkEncoding.size(); i++)
    {
        
    }
}