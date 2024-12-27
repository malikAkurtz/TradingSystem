#include "GenFunctions.h"

using namespace LinearAlgebra;

double calculateMSE(const std::vector<double> &predictions, std::vector<double> &labels)
{
    int num_elements = predictions.size();

    std::vector<double> resultant = subtractVectors(predictions, labels);
    return innerProduct(resultant, resultant) / num_elements;
}

double calculateMSE_Simple(const std::vector<double> &predictions, const std::vector<double> &labels)
{
    std::vector<double> resultant = subtractVectors(predictions, labels);
    return innerProduct(resultant, resultant) / 2;
}

double calculateLogLoss(const std::vector<double> &predictions, const std::vector<double> &labels)
{
    double cumSum = 0;
    const double epsilon = 1e-10;
    for (int i = 0; i < predictions.size(); i++)
    {
        double clipped_prediction = std::max(epsilon, std::min(1 - epsilon, predictions[i]));
        cumSum += (labels[i] * std::log(clipped_prediction)) + ((1 - labels[i]) * std::log((1 - clipped_prediction)));
    }
    return (-1 * (cumSum / predictions.size()));
}



std::vector<double> thresholdFunction(const std::vector<double> &softPredictions, const double &threshhold)
{
    std::vector<double> hardPredictions(softPredictions.size());

    for (int i = 0; i < softPredictions.size(); i++)
    {
        if (softPredictions[i] >= threshhold)
        {
            hardPredictions[i] = 1.0;
        }
        else
        {
            hardPredictions[i] = 0.0;
        }
    }

    return hardPredictions;
}



double calculateMean(const std::vector<double> &v1)
{
    return (accumulateVector(v1) / v1.size());
}

double calculateSTD(const std::vector<double> &v1)
{
    int num_elements = v1.size();
    double mean = calculateMean(v1);

    std::vector<double> mean_vector = createVector(mean, num_elements);

    std::vector<double> normalized = subtractVectors(v1, mean_vector);
    double IP = innerProduct(normalized, normalized);

    return (std::sqrt(IP / num_elements));
}

std::vector<std::vector<double>> normalizeData(const std::vector<std::vector<double>> &dataMatrix)
{
    int num_cols = dataMatrix[0].size();
    int num_rows = dataMatrix.size();
    std::vector<std::vector<double>> normalized_matrix(num_rows, std::vector<double>(num_cols, 0));

    // for every column
    for (int j = 0; j < num_cols; j++)
    {
        std::vector<double> col_to_normalize = getColumn(dataMatrix, j);
        std::vector<double> pre_normalized = subtractVectors(col_to_normalize, createVector(calculateMean(col_to_normalize), num_rows));
        double col_STD = calculateSTD(col_to_normalize);
        if (col_STD == 0)
        {
            throw std::runtime_error("Standard deviation is zero, cannot normalize.");
        }
        std::vector<double> normalized = divideVector(pre_normalized, col_STD);

        updateColumn(normalized_matrix, normalized, j);
    }

    return normalized_matrix;
}


std::vector<std::vector<std::vector<double>>> createBatches(const std::vector<std::vector<double>> &features, int batchSize) {
    std::vector<std::vector<std::vector<double>>> batches;

    int num_samples = features.size();
    int num_batches = (num_samples + batchSize - 1) / batchSize; // Ceiling division to calculate number of batches

    for (int i = 0; i < num_batches; i++) {
        int start_index = i * batchSize;
        int end_index = std::min(start_index + batchSize, num_samples);
        
        // Create a batch from the feature vectors in the current range
        std::vector<std::vector<double>> batch(features.begin() + start_index, features.begin() + end_index);
        batches.push_back(batch);
    }

    return batches;
}

std::pair<std::vector<std::vector<double>>, std::vector<std::vector<double>>> splitData( std::vector<std::vector<double>> data, float split_ratio)
{
    int split_index = split_ratio * data.size();

    std::vector<std::vector<double>> train_data(data.begin(), data.begin() + split_index);
    std::vector<std::vector<double>> test_data(data.begin() + split_index, data.end());

    return {train_data, test_data};
}