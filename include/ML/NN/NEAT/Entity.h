#ifndef ENTITY_H
#define ENTITY_H

#include "Genome.h"
#include "NeuralNet.h"
#include "LossFunctions.h"
#include "EntityID.h"

struct Entity
{
    int id;
    Genome genome;
    double fitness;
    NeuralNet brain;

    Entity(const Genome& genome);

    Genome crossover(Entity &other_parent);

    void evaluateFitness(const std::vector<std::vector<double>> &features_matrix, const std::vector<std::vector<double>> &labels);
};

#endif