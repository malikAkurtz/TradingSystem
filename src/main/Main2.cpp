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

bool DEBUG = false;

int global_innovation_number = 1;
int global_entity_id = 0;


int main()
{
    ////////////////////////////// DATA INPUT //////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    //std::vector<std::vector<double>> data = parseCSV("/Users/malikkurtz/Coding/TradingSystem/data/csv/XRP Historical Data.csv");
    std::vector<std::vector<double>> data = data2;
    

    ////////////////////////////// DATA MODIFICATION //////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    //std::reverse(data.begin(), data.end());

    // convert percentages to just up or down
    // for (int i = 0; i < data.size(); i++)
    // {
    //     data[i][5] = (data[i][5] >= 0) ? true : false;
    // }

    // if we have a very large dataset and only want to test on a small subset
    // float ratio = 0.8;
    // int start_index = data.size() * ratio;
    // std::vector<std::vector<double>> data_shortened(data.begin() + start_index, data.end());
    // data = data_shortened;

    int label_index = 1;

    // Separate labels from data before normalization
    std::vector<std::vector<double>> labels = LinearAlgebra::vector1DtoColumnVector(LinearAlgebra::getColumn(data, label_index));
    LinearAlgebra::deleteColumn(data, label_index);

    // Normalize only the feature columns
    data = normalizeData(data); //******************************* YOU COMMENTED THIS OUT************

    // Reattach labels to the data for splitting
    for (size_t i = 0; i < data.size(); i++) {
        data[i].push_back(labels[i][0]);
    }

    ////////////////////////////// DATA PARTITIONING //////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

    // Split the data into training and validation sets, the second parameter is the ratio of training to validation data
    std::pair<std::vector<std::vector<double>>, std::vector<std::vector<double>>> csv_data = splitData(data, 0.8);

    // Extract features and labels for training and validation
    std::vector<std::vector<double>> X_train = csv_data.first;
    std::vector<std::vector<double>> Y_train = LinearAlgebra::vector1DtoColumnVector(LinearAlgebra::getColumn(X_train, label_index));
    LinearAlgebra::deleteColumn(X_train, label_index);

    std::vector<std::vector<double>> X_val = csv_data.second;
    std::vector<std::vector<double>> Y_val = LinearAlgebra::vector1DtoColumnVector(LinearAlgebra::getColumn(X_val, label_index));
    LinearAlgebra::deleteColumn(X_val, label_index);

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

    ////////////////////////////// BASE GENOME INITIALIZATION //////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    Genome base_genome(num_features, 1);

    std::cout << "Base Genome is:" << std::endl;
    std::cout << base_genome.toString() << std::endl;
    std::cout << "-------------------------------------------------------------------" << std::endl;

    ////////////////////////////// HYPER-PARAMETER SPECIFICATION //////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    int max_generations = 1000;
    int population_size = 100;
    float elite_ratio = 0.2;

    double weight_mutation_rate = 0.8;
    double  add_connection_mutation_rate = 0.4; // 0.05
    double add_node_mutation_rate = 0.3; // 0.03
    std::uniform_real_distribution<> dis(0.0, 1.0);

    ////////////////////////////// BASE POPULATION INITIALIZATION //////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    std::vector<Entity> population;
    population.reserve(population_size);

    for (int i = 0; i < population_size; i++)
    {
        population.emplace_back(Entity(base_genome));
    }

    std::cout << "Base Entity Neural Network toString" << std::endl;
    std::cout << population[0].brain.toString() << std::endl;

    ////////////////////////////// MAIN LOOP //////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
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