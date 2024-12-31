#include "Entity.h"

Entity::Entity(const Genome& genome)
{
    this->genome = genome;
    this->brain = NeuralNet(genome);
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

    offspring.node_genes = most_fit->genome.node_genes;

    auto least_fit_it = least_fit->genome.connection_genes.begin();


    for (const auto& fit_connection_gene : most_fit->genome.connection_genes)
    {
        std::cout << "Most Fit it is at: " << fit_connection_gene.toString() << std::endl;
        std::cout << "Most Fit Parent IN: " << fit_connection_gene.innovation_number << " Least Fit Parent IN: " << least_fit_it->innovation_number << std::endl;
        if (least_fit_it == least_fit->genome.connection_genes.end())
        {
            offspring.connection_genes.push_back(fit_connection_gene);
        }
        else
        {
            if (fit_connection_gene.innovation_number == least_fit_it->innovation_number)
            {
                if (rand() % 2)
                {
                    offspring.connection_genes.push_back(fit_connection_gene);
                }
                else
                {
                    offspring.connection_genes.push_back(*least_fit_it);
                }
                least_fit_it++;
            }
            else if (fit_connection_gene.innovation_number > least_fit_it->innovation_number)
            {
                offspring.connection_genes.push_back(fit_connection_gene);
                least_fit_it++;
            }
            else if (fit_connection_gene.innovation_number < least_fit_it->innovation_number)
            {
                offspring.connection_genes.push_back(fit_connection_gene);
            }
        }
    }



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

    this->fitness = entity_loss;
}