#include <vector>
#include <map>
#include <numeric>
#include <random>
#include <limits>
#include <iostream>
#include <cmath>
#include "Math/LinearAlgebra.h"

class KMeans {
public:
    int k;
    std::map<int, std::vector<std::vector<float>>> cluster_assignments;
    std::map<int, std::vector<float>> cluster_coordinates;

    KMeans(int k) {
        this->k = k;
    }

    std::map<int, std::vector<std::vector<float>>> fit(std::vector<std::vector<float>> data_points_vector) {
        std::random_device rd{};
        std::mt19937 gen{rd()};
        std::normal_distribution<double> distribution{0.0, 1.0};

        int num_features = data_points_vector[0].size();
        std::vector<float> feature_averages(num_features);

        std::vector<std::vector<float>> data_points_vector_T = takeTranspose(data_points_vector);

        for (int i = 0; i < num_features; i++) {
            int sum = std::accumulate(data_points_vector_T[i].begin(), data_points_vector_T[i].end(), 0);
            feature_averages[i] = static_cast<double>(sum) / data_points_vector_T[i].size();
        }

        for (int i = 0; i < k; i++) {
            std::vector<float> cluster_coords(num_features);
            cluster_coordinates.insert({i, cluster_coords});

            for (int j = 0; j < cluster_coords.size(); j++) {
                cluster_coordinates[i][j] = feature_averages[j] + distribution(gen); // random noise
            }
        }

        // Assign each data point to the closest cluster
        for (int i = 0; i < data_points_vector.size(); i++) {
            std::vector<float> data_coord = data_points_vector[i];

            int closest_cluster = 0;
            float dist_closest_cluster = std::numeric_limits<int>::max();

            for (int j = 0; j < k; j++) {
                std::vector<float> cluster_coord = cluster_coordinates[j];
                float dist = calculateNorm(subtractVectors(cluster_coord, data_coord));

                if (dist < dist_closest_cluster) {
                    closest_cluster = j;
                    dist_closest_cluster = dist;
                }
            }

            cluster_assignments[closest_cluster].push_back(data_coord);
        }

        // Iteratively refine the cluster centroids
        bool optimized = false;
        while (!optimized) {
            float cluster_coord_cum_delta = 0;
            // Calculate new cluster centroids
            for (int i = 0; i < k; i++) {
                if (cluster_assignments[i].empty()) {
                continue; 
                }
                std::vector<float> old_cluster_coords = cluster_coordinates[i];
                for (int j = 0; j < num_features; j++) {
                    std::vector<float> feature_vals = takeTranspose(cluster_assignments[i])[j];
                    int sum = std::accumulate(
                        feature_vals.begin(),
                        feature_vals.end(),
                        0
                    );
                    cluster_coordinates[i][j] = sum / cluster_assignments[i].size();
                }
                std::vector<float> new_cluster_coords = cluster_coordinates[i];
                cluster_coord_cum_delta += calculateNorm(subtractVectors(old_cluster_coords, new_cluster_coords));
            }
            float cluster_coord_avg_delta = cluster_coord_cum_delta / k;

            // Reassign points to the closest cluster
            cluster_assignments.clear();
            for (int i = 0; i < data_points_vector.size(); i++) {
                std::vector<float> data_coord = data_points_vector[i];

                int closest_cluster = 0;
                float dist_closest_cluster = std::numeric_limits<int>::max();

                for (int j = 0; j < k; j++) {
                    std::vector<float> cluster_coord = cluster_coordinates[j];
                    float dist = calculateNorm(subtractVectors(cluster_coord, data_coord));

                    if (dist < dist_closest_cluster) {
                        closest_cluster = j;
                        dist_closest_cluster = dist;
                    }
                }

                cluster_assignments[closest_cluster].push_back(data_coord);
            }


            if (cluster_coord_avg_delta < 0.01) {
                optimized = true;
            }
        }

        return cluster_assignments;
    }
};

