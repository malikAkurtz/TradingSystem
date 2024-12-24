#ifndef NEURO_FUNCS
#define NEURO_FUNCS

#include "NeuralNetwork.h"
#include <random>

std::vector<double> uniformCrossover(const NeuralNetwork &p1, const NeuralNetwork &p2);
void mutate(std::vector<double>& networkEncoding, const float& mutation_rate, const float& mutation_range);

#endif