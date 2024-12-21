#include <vector>
#include <fstream>
#include <sstream>
#include <iostream>
#include <cctype> // For std::isdigit
#include <string>

// Helper function to trim quotes
std::string trimQuotes(const std::string& str) {
    if (str.size() >= 2 && str.front() == '"' && str.back() == '"') {
        return str.substr(1, str.size() - 2);
    }
    return str;
}

std::vector<std::string> getCSVHeaders(std::string file_name){
    std::vector<std::string> headers;

    std::ifstream file(file_name);
    if(!file.is_open()) {
        return headers;
    }

    std::string line;

    std::getline(file, line);
    std::stringstream ss(line);

    std::string cell;
    while (std::getline(ss, cell, ',')) {
        headers.push_back(cell);
    }
    
    return headers;
    file.close();
}

std::vector<std::string> getCSVIndices(std::string file_name, int index_col) {
    std::vector<std::string> indices;

    std::ifstream file(file_name);
    if (!file.is_open()) {
        return indices; // return empty if the file can't be opened
    }

    std::string line;

    // Skip headers
    std::getline(file, line);

    // Process each subsequent line
    while (std::getline(file, line)) {
        if (line.empty()) continue;

        std::stringstream ss(line);
        std::string cell;
        int col = 0;

        while (std::getline(ss, cell, ',')) {
            cell = trimQuotes(cell); // Remove surrounding quotes

            if (col == index_col) {
                indices.push_back(cell);
            }

            col++;
        }
    }

    file.close();
    return indices;
}


std::vector<std::vector<double>> parseCSV(std::string file_name) {
    std::vector<std::vector<double>> dataMatrix;

    std::ifstream file(file_name);
    if (!file.is_open()) {
        return dataMatrix; // return empty matrix
    }

    std::string line;

    std::getline(file, line); // skip headers

    while (std::getline(file, line)) {
        if (line.empty()) continue;

        std::stringstream ss(line);
        std::string cell;
        std::vector<double> rowValues;

        int col = 0;
        while (std::getline(ss, cell, ',')) {
            cell = trimQuotes(cell); // Remove surrounding quotes

            if (col == 0) { // Skip the first column (row index)
                col++;
                continue;
            }

            try {
                rowValues.push_back(std::stof(cell));
            } catch (const std::invalid_argument& e) {
            }
            col++;
        }

        if (!rowValues.empty()) {
            dataMatrix.push_back(rowValues);
        }
    }

    file.close();
    return dataMatrix;
}


void toCSV(const std::string &file_name,
           const std::vector<int> &epochs,
           const std::vector<double> &losses,
           const std::vector<double> &gradients) 
{
    // Check for consistent sizes
    if (epochs.size() != losses.size()) {
        std::cerr << "Epochs and losses sizes do not match." << std::endl;
        return;
    }

    std::ofstream file(file_name);
    if (!file.is_open()) {
        std::cerr << "Could not open file for writing: " << file_name << std::endl;
        return;
    }

    // Write header row
    file << "Epoch,Loss,Gradient\n";

    // Write each epoch and its corresponding loss
    for (size_t i = 0; i < epochs.size(); ++i) {
        file << epochs[i] << "," << losses[i] << "," << gradients[i] << "\n";
    }

    file.close();
}

