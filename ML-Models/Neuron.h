#ifndef NEURON_H
#define NEURON_H

#include <vector>

// Neuron Class
class Neuron {
public:
    // Member variable
    std::vector<double> weights;

    // Constructor to initialize weights to a default value
    explicit Neuron(int num_weights);

    // Getter for weights
    std::vector<double> getWeights() const;

    // Setter for weights
    void setWeights(const std::vector<double>& new_weights);
};


///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////



// InputNeuron Class
class InputNeuron {
public:
    // Member variable
    double input;

    // Default constructor
    InputNeuron();

    // Getter for input
    double getInput() const;

    // Setter for input
    void setInput(double new_input);
};

#endif // NEURON_H
