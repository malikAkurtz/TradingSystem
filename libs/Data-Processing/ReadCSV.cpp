#include <vector>
#include <fstream>
#include <sstream>
#include <cctype> // For std::isdigit

// Helper function to trim quotes
std::string trimQuotes(const std::string& str) {
    if (str.size() >= 2 && str.front() == '"' && str.back() == '"') {
        return str.substr(1, str.size() - 2);
    }
    return str;
}

std::vector<std::vector<float>> parseCSV(std::string file_name) {
    std::vector<std::vector<float>> dataMatrix;

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
        std::vector<float> rowValues;

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
