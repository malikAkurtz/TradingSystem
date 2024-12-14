#include <vector>
#include <fstream>
#include <sstream>

std::vector<std::vector<float>> parseCSV(std::string file_name) {
    std::vector<std::vector<float>> dataMatrix;

    std::ifstream file(file_name);

    std::string line;

    std::getline(file, line); // skip headers

    while (std::getline(file, line)) {
        if (line.empty()) {continue;}

        std::stringstream ss(line);

        std::string cell;
        std::vector<float> rowValues;

        while (std::getline(ss, cell, ',')) {
            if (cell == "M") {
                rowValues.push_back(1.0);
            }
            else if (cell == "B") {
                rowValues.push_back(0.0);
            } else {
                rowValues.push_back(std::stof(cell));
            }
            
        }
        dataMatrix.push_back(rowValues);
    }

    file.close();
    return dataMatrix;
}