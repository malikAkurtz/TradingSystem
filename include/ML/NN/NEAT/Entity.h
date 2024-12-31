#ifndef ENTITY_H
#define ENTITY_H

#include "Genome.h"

struct Entity
{
    Genome genome;
    double fitness;

    Genome crossover(Entity &other_parent);
};

#endif