#ifndef CSV_READER
#define CSV_READER

#include <vector>
#include <fstream>
#include <sstream>

std::vector<std::vector<double>> parseCSV(std::string file_name);
std::vector<std::string> getCSVHeaders(std::string file_name);
std::vector<std::string> getCSVIndices(std::string file_name, int index_col);
void toCSV(const std::string &file_name,
           const std::vector<std::vector<double>> &data,
           const std::vector<double> &labels,
           const std::vector<double> &predictions);


#endif