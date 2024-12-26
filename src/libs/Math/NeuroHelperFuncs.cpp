#include "NeuroHelperFuncs.h"


void randomizeEncoding(std::vector<double>& encoding)
{
    static std::mt19937 generator(std::random_device{}());
    std::uniform_real_distribution<double> distribution(-0.5, 0.5);

    for (auto& weight : encoding)
    {
        weight = distribution(generator);
    }
}
std::vector<double> uniformCrossover(const NeuralNetwork& p1, const NeuralNetwork& p2)
{
    
    std::vector<double> p1_encoding = p1.getNetworkEncoding();
    std::vector<double> p2_encoding = p2.getNetworkEncoding();

    printDebug("Parent 1 encoding:");
    printVectorDebug(p1_encoding);
    printDebug("Parent 2 encoding:");
    printVectorDebug(p2_encoding);

    if (p1_encoding.size() != p2_encoding.size())
    {
        throw std::invalid_argument("Cant crossover, network Encodings dont have same structure!");
    }
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

std::vector<double> mutate(std::vector<double>& networkEncoding, const float& mutation_rate)
{
    std::vector<double> mutated = networkEncoding;
    std::random_device rd;
    std::mt19937 gen(rd());

    std::uniform_real_distribution<> scale_dist(0.5, 1.5); // generate random number in between 0.5 and 1.5
    std::uniform_real_distribution<> offset_dist(1, 5);
    std::bernoulli_distribution bernoulli_dist(mutation_rate); 

    for (int i = 0; i < networkEncoding.size(); i++)
    {
        if (bernoulli_dist(gen))
        {
            double random_factor = scale_dist(gen);
            double offset = offset_dist(gen);
            mutated[i] = (mutated[i] * random_factor) + offset;
        }
    }

    return mutated;
}
