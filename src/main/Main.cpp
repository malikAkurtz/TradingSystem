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
#include "ReadCSV.h"
#include "GenFunctions.h"
#include <algorithm>
#include "PopulationID.h"

bool DEBUG = false;

int global_innovation_number = 1;
int global_entity_id = 0;
int global_population_id = 0;

int main()
{
    std::vector<std::vector<double>> data = data3;

    int label_index = 3;


    // Separate labels from data before normalization

    std::vector<std::vector<double>> labels = LinearAlgebra::vector1DtoColumnVector(LinearAlgebra::getColumn(data, label_index));
    LinearAlgebra::deleteColumn(data, label_index);

    //Normalize only the feature columns
    data = normalizeData(data); //******************************* YOU COMMENTED THIS OUT************

    // Reattach labels to the data for splitting
    for (size_t i = 0; i < data.size(); i++) {
        data[i].push_back(labels[i][0]);
    }

    // Split the data into training and validation sets
    auto csv_data = splitData(data, 0.8);

    // Extract features and labels for training and validation
    std::vector<std::vector<double>> X_train = csv_data.first;
    std::vector<std::vector<double>> Y_train = LinearAlgebra::vector1DtoColumnVector(LinearAlgebra::getColumn(X_train, label_index));
    LinearAlgebra::deleteColumn(X_train, label_index);

    std::vector<std::vector<double>> X_val = csv_data.second;
    std::vector<std::vector<double>> Y_val = LinearAlgebra::vector1DtoColumnVector(LinearAlgebra::getColumn(X_val, label_index));
    LinearAlgebra::deleteColumn(X_val, label_index);

    // Neural Network Initialization
    int num_features = X_train[0].size();
    int num_labels = Y_train[0].size();

    printDebug("X_train:");
    printMatrixDebug(X_train);

    printDebug("Y_train:");
    printMatrixDebug(Y_train);

    printDebug("X_val:");
    printMatrixDebug(X_val);

    printDebug("Y_val:");
    printMatrixDebug(Y_val);

    srand(time(0));
    std::random_device rd;
    std::mt19937 gen(rd());

    Genome base_genome(num_features, num_labels);

    std::cout << "Base Genome is:" << std::endl;
    std::cout << base_genome.toString() << std::endl;
    std::cout << "-------------------------------------------------------------------" << std::endl;
    int max_generations = 1000;
    int population_size = 100;
    float elite_ratio = 0.1;
    double speciation_threshold = 3.0;

    double weight_mutation_rate = 0.8;
    double  add_connection_mutation_rate = 0.05;
    double add_node_mutation_rate = 0.03;
    std::uniform_real_distribution<> dis(0.0, 1.0);


    //initialie the base population
    std::map <int, std::vector<std::shared_ptr<Entity>>> this_speciated_population;

    std::vector<Entity> this_population;

    // for use during speciation
    std::map <int, std::vector<std::shared_ptr<Entity>>> prev_speciated_population;

    std::vector<Entity> prev_population;

    for (int i = 0; i < population_size; i++)
    {
        prev_population.emplace_back(Entity(base_genome));
    }

    this_population = prev_population;

    std::cout << "Base Entity Neural Network toString" << std::endl;
    std::cout << this_population[0].brain.toString() << std::endl;

    prev_speciated_population[global_population_id++] = {};
    // initial population is species 1
    for (auto& entity : prev_population)
    {
        prev_speciated_population[0].push_back(std::shared_ptr<Entity>(&entity));
    }

    this_speciated_population = prev_speciated_population;

    // for every generation
    for (int i = 0; i < max_generations; i++)
    {
        std::cout << "------------------BEGINNING GENERATION " << i << "-------------------" << std::endl;

        // group the new population into species based on the prev population
        // for every species in the previous population
        for (const auto& [species_num, entity_members] : prev_speciated_population)
        {
            // get a species representative
            const std::shared_ptr<Entity> species_representative = entity_members[0];
            // for every member in the population
            for (auto& new_entity : this_population)
            {
                // compare compatibility distance
                double compatibility_dist = species_representative->genome.calculateCompatibilityDist(new_entity.genome);
                if (compatibility_dist <= speciation_threshold)
                {
                    this_speciated_population[species_num].push_back(std::shared_ptr<Entity>(&new_entity));
                }
                else
                {
                    this_speciated_population[global_population_id++].push_back(std::shared_ptr<Entity>(&new_entity));
                }
            }
        }

        std::map <int, double> species_cum_fitness;
        double total_fitness = 0;

        for (auto& [species_num, entity_members] : this_speciated_population)
        {
            for (auto& this_entity : entity_members)
            {
                this_entity->evaluateFitness(X_train, Y_train);
                this_entity->fitness /= entity_members.size();
                species_cum_fitness[species_num] += this_entity->fitness;
                total_fitness += this_entity->fitness;
            }
        }

        std::sort(this_population.begin(), this_population.end(), [](const Entity &a, const Entity &b)
                  { return a.fitness > b.fitness; }); 

        for (const auto& [species_num, entity_members] : this_speciated_population)
        {
            std::sort(entity_members.begin(), entity_members.end(), [](const Entity* entity1, const Entity* entity2)
                  { return (entity1->fitness > entity2->fitness); });
        }


        int num_elites = population_size * elite_ratio;
        // std::cout << "Number of elites selected for crossover: " << num_elites << std::endl;
        int offspring_required = population_size - num_elites;
        // std::cout << "Number of offspring required: " << offspring_required << std::endl;

        this_population.erase(this_population.begin() + num_elites, this_population.end());

        for (const auto& [species_num, entity_members] : this_speciated_population)
        {
            int num_offspring = floor((species_cum_fitness[species_num] / total_fitness) * offspring_required);

            for (int i = 0 ; i < num_offspring; i++)
            {
                
            }
        }

        // evaluate the population
        std::cout
            << "--------------START EVALUATING POPULATION FITNESS--------------" << std::endl;
        for (auto &entity : population)
        {
            // std::cout << "----Evalutating Fitness of Entity: " << entity.id << "-----" << std::endl << entity.genome.toString() << std::endl;
            // std::cout << "Neural Network looks like: " << std::endl << entity.brain.toString() << std::endl;
            entity.evaluateFitness(X_train, Y_train);
            std::cout << "Fitness of Entity: " << entity.id << " = " << entity.fitness << std::endl;
        }
        std::cout << "----------------END EVALUATING POPULATION FITNESS--------------" << std::endl;

        std::sort(population.begin(), population.end(), [](const Entity &a, const Entity &b)
                  { return a.fitness > b.fitness; }); 
        
        if (DEBUG)
        {
            std::cout << "Sorted Fitnesses in decreasing order:" << std::endl;
            for (int j = 0; j < population.size(); j++)
            {
                std::cout << "Entity: " << population[j].id << " Fitness: " << population[j].fitness << std::endl;
            }
        }
        
        // select the top 20% for crossover
        int num_elites = population_size * elite_ratio;
        // std::cout << "Number of elites selected for crossover: " << num_elites << std::endl;
        int offspring_required = population_size - num_elites;
        // std::cout << "Number of offspring required: " << offspring_required << std::endl;

        population.erase(population.begin() + num_elites, population.end());

        if (DEBUG)
        {
            std::cout << "Elites For This Generation Are: " << std::endl;
            for (int j = 0; j < population.size(); j++)
            {
                std::cout << "Entity: " << population[j].id << " Fitness: " << population[j].fitness << std::endl;
            }
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
                //std::cout << "Going To Mutate by Changing a Connection" << std::endl;
                offspring_genome.mutateChangeWeight();
            }
            if (dis(gen) < add_connection_mutation_rate)
            {   
                //std::cout << "Going To Mutate by Adding a Connection" << std::endl;
                offspring_genome.mutateAddConnection();
            }
            if (dis(gen) < add_node_mutation_rate)
            {
                //std::cout << "Going To Mutate by Adding a Node" << std::endl;
                offspring_genome.mutateAddNode();
            }

            // std::cout << "New Mutated Genome is: " << offspring_genome.toString() << std::endl;

            population.emplace_back(Entity(offspring_genome));
        }
        std::cout << "------------------FINISHED GENERATION " << i << "-------------------" << std::endl;
    }

    for (auto& entity : population)
    {
        entity.evaluateFitness(X_train, Y_train);
    }
    std::sort(population.begin(), population.end(), [](const Entity &a, const Entity &b)
                { return a.fitness > b.fitness; });

    Entity best_entity = population[0];

    std::vector<std::vector<double>> training_predictions = best_entity.brain.feedForward(X_train);

    std::cout << "~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~" << std::endl;
    std::cout << "Best Entity is: " << best_entity.id << std::endl;
    std::cout << "Has Fitness: " << best_entity.fitness << std::endl;

    std::cout << "Predictions are: " << std::endl;
    printMatrix(training_predictions);

    std::cout << "Final Genome" << std::endl;
    std::cout << best_entity.genome.toString() << std::endl;

    std::cout << "Final Neural Net" << std::endl;
    std::cout << best_entity.brain.toString();

    best_entity.genome.saveToDotFile("genome_graph.dot");

    best_entity.evaluateFitness(X_val, Y_val);
    std::vector<std::vector<double>> validation_predictions = best_entity.brain.feedForward(X_val);

    std::cout << "Fitness on Validation Data: " << best_entity.fitness << std::endl;

    std::cout << "Validation Predictions: " << std::endl;
    printMatrix(validation_predictions);

    return 0;
}