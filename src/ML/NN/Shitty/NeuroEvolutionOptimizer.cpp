#include "NeuroEvolutionOptimizer.h"

using namespace LinearAlgebra;

NeuroEvolutionOptimizer::NeuroEvolutionOptimizer()
{
    this->mutation_rate = 0.01;
    this->population_size = 100;
    this->max_generations = 1000;
    this->loss_function = SQUARRED_ERROR;
}

NeuroEvolutionOptimizer::NeuroEvolutionOptimizer(float mutation_rate, int population_size, int max_generations, LossFunction loss_function) : mutation_rate(mutation_rate), population_size(population_size), max_generations(max_generations), loss_function(loss_function) {};

void NeuroEvolutionOptimizer::fit(NeuralNetwork &this_network, const std::vector<std::vector<double>>& features_matrix, const std::vector<std::vector<double>>& labels)
{
    // this is what were looking for
    std::vector<double> best_encoding; 

    printDebug("Mutation rate is");
    printDebug(this->mutation_rate);

    int num_samples = features_matrix.size();
    printDebug("Number of samples is");
    printDebug(num_samples);

    std::vector<std::vector<double>> labels_T = takeTranspose(labels);
    std::vector<std::vector<double>> features_matrix_T = takeTranspose(features_matrix);
    printDebug("Features tranpose are");
    printMatrixDebug(features_matrix_T);

    printDebug("Labels tranposed are");
    printMatrixDebug(labels_T);

    std::vector<double> base_encoding = this_network.getNetworkEncoding();
    printDebug("Base network encoding is");
    printVectorDebug(base_encoding);

    // Initialize Population
    std::vector<NeuralNetwork> population;
    population.reserve(this->population_size);

    for (int i = 0; i < this->population_size; i++)
    {
        
        std::vector<double> new_member_encoding = base_encoding;
        randomizeEncoding(new_member_encoding);
        printDebug("New member encoding after mutation is:");
        printVectorDebug(new_member_encoding);
        population.emplace_back(this_network, new_member_encoding);

    }

    printDebug("New Member Encodings:");
    for (auto& nn : population)
    {
        printVectorDebug(nn.getNetworkEncoding());
    }

    // for every generation
    for (int g = 0; g < this->max_generations; g++)
    {  
        std::vector<std::pair<double, NeuralNetwork>> population_loss;
        // need to evaluate the population
        // for every network in the population
        for (int i = 0; i < population_size; i++)
        {
            NeuralNetwork &this_NN = population[i];
            
            // perform a forward pass of the entire dataset through this network

            // A matrix where each column is a prediction for that sample from left to right
            std::vector<std::vector<double>> this_NN_outputs = this_NN.feedForward(features_matrix);
            double this_NN_loss = 0;
            for (int j = 0; j < this_NN_outputs[0].size(); j++)
            {
                this_NN_loss += this->calculateLoss(getColumn(this_NN_outputs, j), getColumn(labels_T, j));
            }
            this_NN_loss /= num_samples;
            std::pair<double, NeuralNetwork> NN_x_Loss(this_NN_loss, this_NN);
            printDebug("This encodings loss is:");
            printDebug(this_NN_loss);
            population_loss.push_back(NN_x_Loss);
        }

        // sort the dictionary
        std::sort(population_loss.begin(), population_loss.end(), 
        [](const std::pair<double, NeuralNetwork>& a, const std::pair<double, NeuralNetwork>& b) {
            return a.first < b.first;
        });

        
        best_encoding = population_loss[0].second.getNetworkEncoding();

        printDebug("New best encoding");
        printVectorDebug(best_encoding);
        printDebug("Best encoding loss");
        printDebug(population_loss[0].first);

        // Elitism selection, keeping top 20%
        int num_surviving_networks = this->population_size * 0.2;

        printDebug("Number of surviving networks will be");
        printDebug(num_surviving_networks);

        std::vector<std::pair<double, NeuralNetwork>> elites(population_loss.begin(), population_loss.begin() + num_surviving_networks + 1);

        population.clear();

        for (auto& elite : elites)
        {
            population.push_back(elite.second);
        }

        for (auto& nn : population)
        {
            printDebug("This elite NN's encoding");
            printVectorDebug(nn.getNetworkEncoding());
        }

        // crossover/breeding
        
        
        int children_needed = this->population_size - num_surviving_networks;

        printDebug("Number of children needed is");
        printDebug(children_needed);
        
        for (int i = 0; i < children_needed; i++)
        {
            //pick two random elites
            // printDebug("Random index for parent 1");
            // printDebug(rand() % population_size);
            NeuralNetwork &parent1 = population[rand() % num_surviving_networks];
            NeuralNetwork& parent2 = population[rand() % num_surviving_networks];
            // printDebug("random index for parent 2");
            // printDebug(rand() % population_size);
            std::vector<double> child_encoding = uniformCrossover(parent1, parent2);

            // mutate
            child_encoding = mutate(child_encoding, this->mutation_rate);
            population.emplace_back(this_network, child_encoding);
        }
    }

    printDebug("Best Encoding is");
    printVectorDebug(best_encoding);
    this_network.setEncoding(best_encoding);
}


double NeuroEvolutionOptimizer::calculateLoss(const std::vector<double> &predictions, const std::vector<double> &labels)
{
    if (this->loss_function == SQUARRED_ERROR) {
        return LossFunctions::vectorizedModifiedSquarredError(predictions, labels);
    } else if (this->loss_function == BINARY_CROSS_ENTROPY) {
        return LossFunctions::vectorizedLogLoss(predictions, labels);
    } else {
        throw std::invalid_argument("NO LOSS FUNCTION SELECTED");
    }
}

