#include "NeuralNet.h"
#include "Output.h"
#include <chrono>
#include <thread>
#include <cstdlib>
#include "Entity.h"
#include <random>
#include <TestData.h>
#include "LinearAlgebra.h"
#include <algorithm>

bool DEBUG = false;

int global_innovation_number = 1;
int global_entity_id = 0;
std::map<std::pair<int, int>, int> global_connection_map;

Genome createBaseGenome(int num_features, int num_labels)
{       
    Genome base_genome;
    std::vector<ConnectionGene> connection_genes;
    std::vector<NodeGene> node_genes;

    for (int i = 0; i < num_features; i++)
    {
        node_genes.emplace_back(NodeGene(i + 1, INPUT)); // add the input nodes
    }
    // add the bias node
    node_genes.emplace_back(NodeGene(-1, BIAS));

    // add the output nodes
    for (int i = 0; i < num_labels; i++)
    {
        node_genes.emplace_back(NodeGene(node_genes.size(), OUTPUT));
        for (int j = 0; j < num_features; j++)
        {
            connection_genes.emplace_back(ConnectionGene((j+1), node_genes.back().node_id, static_cast<double>(rand()) / RAND_MAX - 0.5, true, global_innovation_number++));
        }
        // add the bias connection
        connection_genes.emplace_back(ConnectionGene(-1, node_genes.back().node_id, static_cast<double>(rand()) / RAND_MAX - 0.5, true, global_innovation_number++));
    }

    base_genome.node_genes = node_genes;
    base_genome.connection_genes = connection_genes;

    return base_genome;
}

int main()
{
    std::vector<std::vector<double>> data = stockDataTrain;

    std::vector<std::vector<double>> labels = LinearAlgebra::vector1DtoColumnVector(LinearAlgebra::getColumn(data, 6));
    // std::vector<std::vector<double>> other_label = LinearAlgebra::vector1DtoColumnVector(LinearAlgebra::getColumn(data, 3));
    //LinearAlgebra::addColumn(labels, LinearAlgebra::columnVectortoVector1D(other_label));

    std::cout << "Labels are: " << std::endl;
    printMatrix(labels);


    LinearAlgebra::deleteColumn(data, 6);
    // LinearAlgebra::deleteColumn(data, 2);

    
    std::vector<std::vector<double>> features_matrix = data;
    std::cout << "Features Matrix is: " << std::endl;
    printMatrix(features_matrix);


    srand(time(0));
    std::random_device rd;
    std::mt19937 gen(rd());

    Genome base_genome = createBaseGenome(features_matrix[0].size(), labels[0].size());

    std::cout << "Base Genome is:" << std::endl;
    std::cout << base_genome.toString() << std::endl;
    std::cout << "-------------------------------------------------------------------" << std::endl;
    int max_generations = 1000;
    int population_size = 100;
    float elite_ratio = 0.2;

    double weight_mutation_rate = 0.8;
    double  add_connection_mutation_rate = 0.05;
    double add_node_mutation_rate = 0.03;
    std::uniform_real_distribution<> dis(0.0, 1.0);


    //initialie the base population
    std::vector<Entity> population;
    population.reserve(population_size);

    for (int i = 0; i < population_size; i++)
    {
        population.emplace_back(Entity(base_genome));
    }


    std::cout << "Base Entity Neural Network toString" << std::endl;
    std::cout << population[0].brain.toString() << std::endl;

    // for every generation
    for (int i = 0; i < max_generations; i++)
    {
        std::cout << "------------------BEGINNING GENERATION " << i << "-------------------" << std::endl;
        // evaluate the population
        std::cout << "--------------START EVALUATING POPULATION FITNESS--------------" << std::endl;
        for (auto &entity : population)
        {
            // std::cout << "----Evalutating Fitness of Entity: " << entity.id << "-----" << std::endl << entity.genome.toString() << std::endl;
            // std::cout << "Neural Network looks like: " << std::endl << entity.brain.toString() << std::endl;
            entity.evaluateFitness(features_matrix, labels);
        }
        std::cout << "----------------END EVALUATING POPULATION FITNESS--------------" << std::endl;

        std::sort(population.begin(), population.end(), [](const Entity &a, const Entity &b)
                  { return a.fitness > b.fitness; }); 
        
        std::cout << "Sorted Fitnesses in decreasing order:" << std::endl;
        for (int j = 0; j < population.size(); j++)
        {
            std::cout << "Entity: " << population[j].id << " Fitness: " << population[j].fitness << std::endl;
        }
        // select the top 20% for crossover
        int num_elites = population_size * elite_ratio;
        // std::cout << "Number of elites selected for crossover: " << num_elites << std::endl;
        int offspring_required = population_size - num_elites;
        // std::cout << "Number of offspring required: " << offspring_required << std::endl;

        population.erase(population.begin() + num_elites, population.end());
        std::cout << "Elites For This Generation Are: " << std::endl;
        for (int j = 0; j < population.size(); j++)
        {
            std::cout << "Entity: " << population[j].id << " Fitness: " << population[j].fitness << std::endl;
        }

        // perform crossover
        for (int j = 0; j < offspring_required; j++)
        {
            
            int random_elite_index1 = rand() % num_elites;
            int random_elite_index2 = rand() % num_elites;

            while (random_elite_index2 == random_elite_index1)
            {
                random_elite_index2 = rand() % num_elites;
            }
            
            Entity& random_elite1 = population[random_elite_index1];
            Entity& random_elite2 = population[random_elite_index2];

            Genome offspring_genome = random_elite1.crossover(random_elite2);
            // std::cout << "New Offspring Genome is: " << offspring_genome.toString() << std::endl;

            //perform mutations
            if (dis(gen) < weight_mutation_rate)
            {   
                std::cout << "Going To Mutate by Changing a Connection" << std::endl;
                offspring_genome.mutateChangeWeight();
            }
            if (dis(gen) < add_connection_mutation_rate)
            {   
                std::cout << "Going To Mutate by Adding a Connection" << std::endl;
                offspring_genome.mutateAddConnection();
            }
            if (dis(gen) < add_node_mutation_rate)
            {
                std::cout << "Going To Mutate by Adding a Node" << std::endl;
                offspring_genome.mutateAddNode();
            }

            // std::cout << "New Mutated Genome is: " << offspring_genome.toString() << std::endl;

            population.emplace_back(Entity(offspring_genome));
        }
        std::cout << "------------------FINISHED GENERATION " << i << "-------------------" << std::endl;
    }

    for (auto& entity : population)
    {
        entity.evaluateFitness(features_matrix, labels);
    }
    std::sort(population.begin(), population.end(), [](const Entity &a, const Entity &b)
                { return a.fitness > b.fitness; });

    Entity best_entity = population[0];

    std::vector<std::vector<double>> best_predictions = best_entity.brain.feedForward(features_matrix);

    std::cout << "~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~" << std::endl;
    std::cout << "Best Entity is: " << best_entity.id << std::endl;
    std::cout << "Has Fitness: " << best_entity.fitness << std::endl;

    std::cout << "Predictions are: " << std::endl;
    printMatrix(best_predictions);

    std::cout << "Final Genome" << std::endl;
    std::cout << best_entity.genome.toString() << std::endl;

    std::cout << "Final Neural Net" << std::endl;
    std::cout << best_entity.brain.toString();

    best_entity.genome.saveToDotFile("genome_graph.dot");

    return 0;
}