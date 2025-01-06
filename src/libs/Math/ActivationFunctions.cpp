#include "ActivationFunctions.h"

namespace ActivationFunctions
{
    double ReLU(double value)
    {
        if (value >= 0) 
        {
            return value;
        }
        else
        {
            return 0;
        }
    }
    std::vector<std::vector<double>> matrix_ReLU(const std::vector<std::vector<double>> &v1)
    {
        int num_rows = v1.size();
        int num_cols = v1[0].size();

        std::vector<std::vector<double>> resultant(num_rows, std::vector<double>(num_cols));
        
        for (int i = 0; i < num_rows;i++){
            for (int j = 0; j < num_cols; j++) {
                if (v1[i][j] >= 0) {
                    resultant[i][j] = v1[i][j];
                } else {
                    resultant[i][j] = 0;
                }
            }
        }

        return resultant;
    }

    std::vector<double> vectorReLU(const std::vector<double>& v1)
    {
        std::vector<double> resultant = v1;
        for (auto& value : resultant)
        {
            if (value < 0)
            {
                value = 0;
            }
        }
        return resultant;
    }

    std::vector<std::vector<double>> matrix_d_ReLU(const std::vector<std::vector<double>>& v1)
    {
        int num_rows = v1.size();
        int num_cols = v1[0].size();

        std::vector<std::vector<double>> resultant(num_rows, std::vector<double>(num_cols));
        
        for (int i = 0; i < num_rows;i++){
            for (int j = 0; j < num_cols; j++) {
                if (v1[i][j] >= 0) {
                    resultant[i][j] = 1;
                } else{
                    resultant[i][j] = 0;
                }
            }
        }

        return resultant;
    }


    std::vector<std::vector<double>> matrix_sigmoid(const std::vector<std::vector<double>> &v1)
    {
        int num_rows = v1.size();
        int num_cols = v1[0].size();

        std::vector<std::vector<double>> resultant(num_rows, std::vector<double>(num_cols));

        for (int i = 0; i < num_rows; i++) {
            for (int j = 0; j < num_cols; j++) {
                resultant[i][j] = 1.0 / (1.0 + std::exp(-v1[i][j]));
            }
        }

        return resultant;
    }

    std::vector<double> vectorSigmoid(const std::vector<double>& v1)
    {
        std::vector<double> resultant = v1;
        for (auto& value : resultant)
        {
            value = 1.0 / (1.0 + std::exp(-value));
        }

        return resultant;
    }

    double sigmoid_single(const double &value)
    {
        if (value >= 0)
        {
            return 1.0 / (1.0 + std::exp(-value));
        }
        else
        {
            double exp_val = std::exp(value);
            return exp_val / (1.0 + exp_val);
        }
    }


    std::vector<std::vector<double>> matrix_d_sigmoid(const std::vector<std::vector<double>> &v1)
    {
        int num_rows = v1.size();
        int num_cols = v1[0].size();

        std::vector<std::vector<double>> resultant(num_rows, std::vector<double>(num_cols));

        for (int i = 0; i < num_rows; i++) {
            for (int j = 0; j < num_cols; j++) {
                double sig = sigmoid_single(v1[i][j]);
                resultant[i][j] = sig * (1 - sig);
            }
        }

        return resultant;
    }


    std::vector<double> vectorTanh(const std::vector<double> &v1)
    {
        std::vector<double> resultant = v1;
        for (auto& value : resultant)
        {
            value = (exp(value) - exp(-value)) / (exp(value) + exp(-value));
        }

        return resultant;
    }
}
