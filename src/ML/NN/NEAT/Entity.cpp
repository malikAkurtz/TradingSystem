#include "Entity.h"

Genome Entity::crossover(Entity &other_parent)
{

    std::vector<std::pair<ConnectionGene, ConnectionGene>> pairs;

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

    //auto most_fit_it = most_fit->genome.connection_genes.begin();
    auto least_fit_it = least_fit->genome.connection_genes.begin();


    std::cout << "Final Fit it is at: " << most_fit->genome.connection_genes.end()->toString() << std::endl;
    // while we have not fully processed both connection sequences
    for (const auto& fit_connection_gene : most_fit->genome.connection_genes)
    {
        std::cout << "Most Fit it is at: " << fit_connection_gene.toString() << std::endl;
        std::cout << "Most Fit Parent IN: " << fit_connection_gene.innovation_number << " Least Fit Parent IN: " << least_fit_it->innovation_number << std::endl;
        // if the sequence matches, inherit a random gene
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

    return offspring;

}