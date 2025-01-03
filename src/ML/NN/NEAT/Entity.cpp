#include "Entity.h"


Entity::Entity(const Genome& genome)
{
    this->genome = genome;
    this->brain = NeuralNet(genome);
    this->id = global_entity_id;
    global_entity_id++;
}

Genome Entity::crossover(Entity &other_parent)
{

    Genome offspring;
    Entity* most_fit;
    Entity* least_fit;

    if (this->fitness > other_parent.fitness)
    {
        most_fit = this;
        least_fit = &other_parent;
    }
    else
    {
        most_fit = &other_parent;
        least_fit = this;
    }

    // offspring inherits topology of most fit parent (https://ai.stackexchange.com/questions/9667/using-neat-will-the-child-of-two-parent-genomes-always-have-the-same-structure)
    offspring.node_genes = most_fit->genome.node_genes;

    std::vector<ConnectionGene> most_fit_sorted = most_fit->genome.connection_genes;
    std::vector<ConnectionGene> least_fit_sorted = least_fit->genome.connection_genes;

    std::sort(most_fit_sorted.begin(), most_fit_sorted.end(), [](const ConnectionGene &a, const ConnectionGene &b)
              { return a.innovation_number < b.innovation_number; });

    std::sort(least_fit_sorted.begin(), least_fit_sorted.end(), [](const ConnectionGene &a, const ConnectionGene &b)
              { return a.innovation_number < b.innovation_number; });


    auto most_fit_it = most_fit_sorted.begin();
    auto least_fit_it = least_fit_sorted.begin();

    while (most_fit_it != most_fit_sorted.end() &&
            least_fit_it != least_fit_sorted.end())
    {
        if (most_fit_it->innovation_number == least_fit_it->innovation_number)
        {

            if (rand() % 2)
            {
                offspring.connection_genes.push_back(*most_fit_it);
            }
            else
            {
                offspring.connection_genes.push_back(*least_fit_it);
            }

            if (!most_fit_it->enabled || !least_fit_it->enabled)
            {   
                if ((rand() % 4) == 0)
                {
                    offspring.connection_genes.back().enabled = false;
                }
            }

            most_fit_it++;
            least_fit_it++;
        }
        else if (most_fit_it->innovation_number < least_fit_it->innovation_number)
        {
            offspring.connection_genes.push_back(*most_fit_it);
            most_fit_it++;
        }
        else 
        {
            least_fit_it++;
        }
    }

    while (most_fit_it != most_fit_sorted.end())
    {
        offspring.connection_genes.push_back(*most_fit_it);
        most_fit_it++;
    }
    // if we didnt get to the end of least fit, just drop them they dont matter


    debugMessage("crossover", "Performed Crossover Between Entities: " + std::to_string(this->id) + ", " + std::to_string(other_parent.id) + + " To Produce: " + offspring.toString() + " For Future Entity: " + std::to_string(global_innovation_number));

    return offspring;
}

void Entity::evaluateFitness(const std::vector<std::vector<double>> &features_matrix, const std::vector<std::vector<double>> &labels)
{
    int num_samples = features_matrix.size();

    std::vector<std::vector<double>> entity_outputs = this->brain.feedForward(features_matrix);

    double cumError = 0;

    for (int i = 0; i < num_samples; i++)
    {
        cumError += LossFunctions::vectorizedModifiedSquarredError(entity_outputs[i], labels[i]);
    }

    double entity_loss = cumError / num_samples;
    debugMessage("evaluateFitness", "Entity: " + std::to_string(this->id) + " Fitness: " + std::to_string(-entity_loss));
    
    this->fitness = -entity_loss; // negative because we want higher to be better
}