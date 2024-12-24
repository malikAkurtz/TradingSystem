#ifndef CSV_READER
#define CSV_READER

#include <vector>
#include <fstream>
#include <sstream>
#include <iostream>
#include <cctype> // For std::isdigit
#include <string>

std::vector<std::vector<double>> parseCSV(std::string file_name);
std::vector<std::string> getCSVHeaders(std::string file_name);
std::vector<std::string> getCSVIndices(std::string file_name, int index_col);
void toCSV(const std::string &file_name,
           const std::vector<int> &epochs,
           const std::vector<double> &losses,
           const std::vector<double> &gradients);

#endif