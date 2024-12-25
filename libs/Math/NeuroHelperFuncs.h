#ifndef NEURO_FUNCS
#define NEURO_FUNCS

#include "NeuralNetwork.h"
#include <random>

void randomizeEncoding(std::vector<double> &encoding);
std::vector<double> uniformCrossover(const NeuralNetwork &p1, const NeuralNetwork &p2);
std::vector<double> mutate(std::vector<double> &networkEncoding, const float &mutation_rate);


#endif