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
           const std::vector<std::vector<double>> &data,
           const std::vector<double> &labels,
           const std::vector<double> &predictions) 
{
    // Check for consistent sizes
    size_t num_rows = data.size();
    if (num_rows == 0) {
        std::cerr << "Data is empty, nothing to write." << std::endl;
        return;
    }

    size_t num_cols = data[0].size();

    if (labels.size() != num_rows || predictions.size() != num_rows) {
        std::cerr << "Labels or predictions size does not match data rows." << std::endl;
        return;
    }

    std::ofstream file(file_name);
    if (!file.is_open()) {
        std::cerr << "Could not open file for writing: " << file_name << std::endl;
        return;
    }

    // Write header row (adjust as needed)
    for (size_t c = 0; c < num_cols; ++c) {
        file << "F" << c << ",";
    }
    file << "Label,Prediction\n";

    // Write each row of data, then label, then prediction
    for (size_t r = 0; r < num_rows; ++r) {
        for (size_t c = 0; c < num_cols; ++c) {
            file << data[r][c] << ((c == num_cols - 1) ? ',' : ',');
        }
        file << labels[r] << "," << predictions[r] << "\n";
    }

    file.close();
}

