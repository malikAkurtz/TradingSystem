#ifndef CSV_READER
#define CSV_READER

#include <vector>
#include <fstream>
#include <sstream>

std::vector<std::vector<float>> parseCSV(std::string file_name);
std::vector<std::string> getCSVHeaders(std::string file_name);
std::vector<std::string> getCSVIndices(std::string file_name, int index_col);
void toCSV(const std::string &file_name,
           const std::vector<std::vector<float>> &data,
           const std::vector<float> &labels,
           const std::vector<float> &predictions);


#endif