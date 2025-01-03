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

bool DEBUG = true;

int global_innovation_number = 0;
int global_entity_id = 0;

int main()
{
    std::vector<std::vector<double>> data = data2;

    std::vector<std::vector<double>> labels = LinearAlgebra::vector1DtoColumnVector(LinearAlgebra::getColumn(data, 1));
    std::cout << "Labels are: " << std::endl;
    printMatrix(labels);

    LinearAlgebra::deleteColumn(data, 1);

    std::vector<std::vector<double>> features_matrix = data;
    std::cout << "Features Matrix is: " << std::endl;
    printMatrix(features_matrix);



    srand(time(0));
    std::random_device rd;
    std::mt19937 gen(rd());

    NodeGene ng1(1, INPUT);
    // NodeGene ng2(2, INPUT);
    NodeGene bias(-1, BIAS);
    NodeGene ng2(2, OUTPUT);

    ConnectionGene cg1(1, 2, 0.2, true, 1);
    global_innovation_number++;
    ConnectionGene bias_conn(-1, 2, 1, true, 2);
    global_innovation_number++;
    // ConnectionGene cg2(2, 3, 0.1, true, 2);
    // global_innovation_number++;

    std::vector<ConnectionGene> connection_genes = {cg1, bias_conn};
    std::vector<NodeGene> node_genes = {ng1, ng2, bias};

    
    Genome base_genome(connection_genes, node_genes);
    std::cout << "Base Genome is:" << std::endl;
    std::cout << base_genome.toString() << std::endl;
    std::cout << "-------------------------------------------------------------------" << std::endl;
    int max_generations = 100;
    int population_size = 100;
    float elite_ratio = 0.2;

    double weight_mutation_rate = 0.99;
    double  add_connection_mutation_rate = 0.2;
    double add_node_mutation_rate = 0.1;
    std::uniform_real_distribution<> dis(0.0, 1.0);


    //initialie the base population
    std::vector<Entity> population;
    population.reserve(population_size);

    for (int i = 0; i < population_size; i++)
    {
        population.emplace_back(Entity(base_genome));
    }


    std::cout << "Base Entity Neural Network toString" << std::endl;
    std::cout << population[9].brain.toString() << std::endl;

    // for every generation
    for (int i = 0; i < max_generations; i++)
    {
        std::cout << "------------------BEGINNING GENERATION " << i << "-------------------" << std::endl;
        // evaluate the population
        std::cout << "--------------START EVALUATING POPULATION FITNESS--------------" << std::endl;
        for (auto &entity : population)
        {
            std::cout << "----Evalutating Fitness of Entity: " << entity.id << "-----" << std::endl << entity.genome.toString() << std::endl;
            std::cout << "Neural Network looks like: " << std::endl << entity.brain.toString() << std::endl;
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
                offspring_genome.mutateChangeWeight();
            }
            // else if (dis(gen) < add_connection_mutation_rate)
            // {
            //     offspring_genome.mutateAddConnection();
            // }
            if (dis(gen) < add_node_mutation_rate)
            {
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