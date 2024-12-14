#include "ML-Models/k_means_clustering.cpp"
#include <iostream>


int main() {
    std::vector<std::vector<float>> data = {
        {1.0, 1.5},
        {1.5, 2.0},
        {2.0, 1.0},
        {6.0, 7.0},
        {6.5, 8.0},
        {7.0, 7.5},
        {8.0, 1.0},
        {8.5, 2.0},
        {9.0, 0.5}
    };

    KMeans KM(3);

    std::map<int, std::vector<std::vector<float>>> clusters = KM.fit(data);

    // Print cluster assignments
    std::cout << "Cluster Assignments:" << std::endl;
    for (const auto& cluster : clusters) {
        std::cout << "Cluster " << cluster.first << ":" << std::endl;
        for (const auto& point : cluster.second) {
            std::cout << "(";
            for (size_t i = 0; i < point.size(); i++) {
                std::cout << point[i];
                if (i != point.size() - 1) std::cout << ", ";
            }
            std::cout << ")" << std::endl;
        }
        std::cout << std::endl;
    }

    // Print cluster centroids
    std::cout << "Cluster Centroids:" << std::endl;
    for (const auto& coord : KM.cluster_coordinates) {
        std::cout << "Cluster " << coord.first << ": (";
        for (size_t i = 0; i < coord.second.size(); i++) {
            std::cout << coord.second[i];
            if (i != coord.second.size() - 1) std::cout << ", ";
        }
        std::cout << ")" << std::endl;
    }


    return 0;
}
