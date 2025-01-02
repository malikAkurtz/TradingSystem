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


    NodeGene ng1(1, INPUT);
    NodeGene ng2(1, INPUT);
    NodeGene ng3(2, OUTPUT);

    ConnectionGene cg1(1, 2, 0.2, true, 1);
    global_innovation_number++;

    Genome base_genome({cg1}, {ng1, ng2});

    std::cout << "Before Mutation" << std::endl;

    std::cout << base_genome.toString() << std::endl;

    base_genome.mutateAddNode();

    std::cout << "After Mutation" << std::endl;

    std::cout << base_genome.toString() << std::endl;

    base_genome.mutateAddNode();

    std::cout << base_genome.toString() << std::endl;
    return 0;
}