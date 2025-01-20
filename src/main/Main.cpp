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
    int max_generations = 100;
    int population_size = 100;
    float elite_ratio = 0.2;
    double speciation_threshold = 3.0;

    double weight_mutation_rate = 0.8;
    double  add_connection_mutation_rate = 0.05;
    double add_node_mutation_rate = 0.03;
    std::uniform_real_distribution<> dis(0.0, 1.0);


    //initialize the base population
    std::map <int, std::vector<std::shared_ptr<Entity>>> this_speciated_population;

    std::vector<std::shared_ptr<Entity>> this_population;

    // for use during speciation
    std::map <int, std::vector<std::shared_ptr<Entity>>> prev_speciated_population;

    std::vector<std::shared_ptr<Entity>> prev_population;

    for (int i = 0; i < population_size; i++)
    {
        prev_population.emplace_back(std::make_shared<Entity>(base_genome));
    }

    this_population = prev_population;

    std::cout << "Base Entity Neural Network toString" << std::endl;
    std::cout << this_population[0]->brain.toString() << std::endl;

    prev_speciated_population[global_population_id++] = {};
    // initial population is species 0
    for (std::shared_ptr<Entity>& entity_ptr : prev_population)
    {
        prev_speciated_population[0].push_back(entity_ptr);
    }

    std::cout << "Default Initial Population: " << std::endl;
        for (const auto &[species_num, entity_members] : prev_speciated_population)
        {
            std::cout << "-------------Species " << species_num << "---------------" << std::endl;
            for (const auto &member : entity_members)
            {
                std::cout << member->id << std::endl;
            }
        }

    // for every generation
    for (int i = 0; i < max_generations; i++)
    {
        std::cout << "------------------BEGINNING GENERATION " << i << "-------------------" << std::endl;
        this_speciated_population.clear();

        // group the new population into species based on the prev population
        std::map<std::shared_ptr<Entity>, bool> already_speciated;
        for (const auto& entity_ptr : this_population)
        {
            already_speciated[entity_ptr] = 0; // not speciated by default
        }
        
        // for every species in the previous population
        for (const auto& [species_num, entity_members] : prev_speciated_population)
        {
            std::cout << "Speciating into Species Bucket: " << species_num << std::endl;
            // get a species representative
            const std::shared_ptr<Entity> species_representative = entity_members[0];
            std::cout << "Species rep is: " << species_representative->id << std::endl;
            // for every member in the new, post-mutated population
            for (auto& new_entity : this_population)
            {
                std::cout << "Classifying Entity: " << new_entity->id << std::endl;
                if (already_speciated[new_entity])
                {
                    std::cout << "Entity has Already Been Classified." << std::endl;
                    continue;
                }
                // compare compatibility distance
                double compatibility_dist = species_representative->genome.calculateCompatibilityDist(new_entity->genome);

                std::cout << "Compatibility Distance is: " << compatibility_dist << std::endl;
                // if the compatibility distance is less than the speciation threshold
                if (compatibility_dist <= speciation_threshold)
                {   
                    // then push a pointer to the entity into the map with the current species being compared
                    std::cout << "Classifying Entity: " << new_entity->id << " into Species: " << species_num << std::endl;
                    this_speciated_population[species_num].push_back(new_entity);
                }
                else
                {
                    // otherwise we need to create a new species
                    std::cout << "Classifying Entity: " << new_entity->id << " into New Species: " << std::endl;
                    this_speciated_population[global_population_id++].push_back(new_entity);
                }

                already_speciated[new_entity] = 1;
            }
        }

        // if (DEBUG)
        // {
        std::cout << "New Speciated Population: " << std::endl;
        for (const auto &[species_num, entity_members] : this_speciated_population)
        {
            std::cout << "-------------Species " << species_num << "---------------" << std::endl;
            for (const auto &member : entity_members)
            {
                std::cout << member->id << std::endl;
            }
        }
        // }

        std::map <int, double> species_cum_fitness;
        double total_fitness = 0;

        // for every species in the new, speciated population
        for (auto& [species_num, entity_members] : this_speciated_population)
        {
            // for every member in the species
            for (auto& this_entity : entity_members)
            {
                // evaluate the entities shared fitness
                this_entity->evaluateFitness(X_train, Y_train);
                this_entity->fitness /= entity_members.size();
                species_cum_fitness[species_num] += this_entity->fitness;
                total_fitness += this_entity->fitness;
            }
        }

        // sort the entities by their fitness
        std::sort(this_population.begin(), this_population.end(),
          [](const std::shared_ptr<Entity> &a, const std::shared_ptr<Entity> &b) {
              return a->fitness > b->fitness;
          });


        // sort each species by each members fitness
        for (auto& [species_num, entity_members] : this_speciated_population)
        {
            std::sort(entity_members.begin(), entity_members.end(),
          [](const std::shared_ptr<Entity> &entity1, const std::shared_ptr<Entity> &entity2) {
              return entity1->fitness > entity2->fitness;
          });

        }

        int num_elites = population_size * elite_ratio;
        // std::cout << "Number of elites selected for crossover: " << num_elites << std::endl;
        int offspring_required = population_size - num_elites;
        // std::cout << "Number of offspring required: " << offspring_required << std::endl;
        std::vector<std::shared_ptr<Entity>> next_generation = {};

        // for every species in the new population
        for (auto& [species_num, entity_members] : this_speciated_population)
        {
            // calculate the number of elites in this species
            int num_species_elites = floor(elite_ratio * entity_members.size());
            // calculate the number of offspring that this species gets to produce
            int num_offspring = floor((species_cum_fitness[species_num] / total_fitness) * offspring_required);

            // nullify (erase) other members from the species
            entity_members.erase(entity_members.begin() + num_species_elites, entity_members.end());

            // want to create offspring out of the elites in this species
            for (int i = 0 ; i < num_offspring; i++)
            {
                Genome offspring_genome;
                // if theres only one elite
                if (entity_members.size() == 1)
                {   
                    // just add copies of the elite to the species
                    offspring_genome = entity_members[0]->genome;
                }
                else
                {
                    int parent1_index = rand() % entity_members.size();
                    int parent2_index = rand() % entity_members.size();

                    while (parent1_index == parent2_index)
                    {
                        parent2_index = rand() % entity_members.size();
                    }

                    std::shared_ptr<Entity> parent1 = entity_members[parent1_index];
                    std::shared_ptr<Entity> parent2 = entity_members[parent2_index];

                    offspring_genome = parent1->crossover(*parent2);
                }
                

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

                next_generation.emplace_back(std::make_shared<Entity>(offspring_genome));
            }
        }
        this_population = next_generation;
        prev_speciated_population = this_speciated_population;
        prev_population = this_population;

        
        std::cout << "------------------FINISHED GENERATION " << i << "-------------------" << std::endl;
    }

    for (auto& entity : this_population)
    {
        entity->evaluateFitness(X_train, Y_train);
    }
    std::sort(this_population.begin(), this_population.end(), [](const std::shared_ptr<Entity> &a, const std::shared_ptr<Entity> &b)
                { return a->fitness > b->fitness; });

    std::shared_ptr<Entity> best_entity = this_population[0];

    std::vector<std::vector<double>> training_predictions = best_entity->brain.feedForward(X_train);

    std::cout << "~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~" << std::endl;
    std::cout << "Best Entity is: " << best_entity->id << std::endl;
    std::cout << "Has Fitness: " << best_entity->fitness << std::endl;

    std::cout << "Predictions are: " << std::endl;
    printMatrix(training_predictions);

    std::cout << "Final Genome" << std::endl;
    std::cout << best_entity->genome.toString() << std::endl;

    std::cout << "Final Neural Net" << std::endl;
    std::cout << best_entity->brain.toString();

    best_entity->genome.saveToDotFile("genome_graph.dot");

    best_entity->evaluateFitness(X_val, Y_val);
    std::vector<std::vector<double>> validation_predictions = best_entity->brain.feedForward(X_val);

    std::cout << "Fitness on Validation Data: " << best_entity->fitness << std::endl;

    std::cout << "Validation Predictions: " << std::endl;
    printMatrix(validation_predictions);

    return 0;
}