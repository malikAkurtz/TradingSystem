import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs
import pandas as pd

counter = 100

class KMeans():
    def __init__(self, k):
        self.num_clusters = k
        self.cluster_dict = {}
        self.centroid_mean_diff_list = []
        self.data = None
        self.num_features = None
        self.df = False

    def fit(self, data_coords):
        # If it is a dataframe
        if isinstance(data_coords, pd.DataFrame):
            data_coords_df = data_coords.copy()
            data_coords = data_coords.values
            data_coords_df["cluster"] = None
            self.df = True
        # If it is not a dataframe, make it one
        else:
            data_coords_df = pd.DataFrame(data_coords)
            data_coords_df["cluster"] = None
            self.df = True

        self.data = data_coords
        self.num_features = self.data.shape[1]
        variance_clusters_dict = {}
        
        for m in range(3):
            self.cluster_dict = {}
            for cluster in range(self.num_clusters):
                cluster_coordinates = []
                for i in range(self.num_features):
                    feature_column_mean = np.mean(self.data[:, i])
                    random_noise = np.random.normal(0, 0.1)
                    cluster_coordinates.append(feature_column_mean + random_noise)
                self.cluster_dict[cluster] = [np.array(cluster_coordinates), []]

            plt.ion()
            fig = plt.figure(figsize=(12, 8))
            if self.num_features == 2:
                ax1 = fig.add_subplot(111)
            elif self.num_features == 3:
                ax1 = fig.add_subplot(111, projection='3d')

            while True:
                if self.num_features <= 3:
                    self.live_plotting(ax1)
                
                for i in list(self.cluster_dict.keys()):
                    self.cluster_dict[i][1] = []
                
                self.centroid_mean_diff_list = []

                for j, data_coord in enumerate(data_coords):
                    closest_cluster_distance = float('inf')
                    closest_cluster = None

                    for i in list(self.cluster_dict.keys()):
                        cluster_coord = self.cluster_dict[i][0]
                        distance = np.linalg.norm(cluster_coord - data_coord)
                        if distance < closest_cluster_distance:
                            closest_cluster_distance = distance
                            closest_cluster = i

                    self.cluster_dict[closest_cluster][1].append(data_coord)
                    if self.df:
                        data_coords_df.at[j, "cluster"] = closest_cluster

                converged = True
                for i in list(self.cluster_dict.keys()):
                    old_centroid_mean = np.array(self.cluster_dict[i][0])
                    if len(self.cluster_dict[i][1]) > 0:
                        self.cluster_dict[i][0] = np.mean(self.cluster_dict[i][1], axis=0)
                    new_centroid_mean = np.array(self.cluster_dict[i][0])
                    mean_change = np.linalg.norm(new_centroid_mean - old_centroid_mean)
                    self.centroid_mean_diff_list.append(mean_change)
                    if mean_change > 0.01:
                        converged = False

                if converged:
                    break

            variance_sum_for_cluster = np.sum(list(self.calculate_cluster_variance().values()))
            variance_clusters_dict[m] = [variance_sum_for_cluster, data_coords_df['cluster'].values.copy()]
        
        print("DONE")
        print(variance_clusters_dict)
        
        best_cluster_index = None
        lowest_variance = float('inf')
        for cluster_i in variance_clusters_dict.keys():
            if variance_clusters_dict[cluster_i][0] < lowest_variance:
                best_cluster_index = cluster_i
                lowest_variance = variance_clusters_dict[cluster_i][0]

        return variance_clusters_dict[best_cluster_index][1]

    def calculate_cluster_variance(self):
        variance_dict = {}
        for cluster in range(self.num_clusters):
            cluster_points = np.array(self.cluster_dict[cluster][1])
            centroid = self.cluster_dict[cluster][0]
            if len(cluster_points) > 0:
                variance_dict[cluster] = np.mean(np.linalg.norm(cluster_points - centroid, axis=1) ** 2)
            else:
                variance_dict[cluster] = 0
        return variance_dict

    def live_plotting(self, ax1):
        ax1.cla()
        colors = plt.cm.get_cmap('viridis', self.num_clusters)

        if self.num_features == 2:
            for i in list(self.cluster_dict.keys()):
                cluster_points = np.array(self.cluster_dict[i][1])
                if len(cluster_points) > 0:
                    ax1.scatter(cluster_points[:, 0], cluster_points[:, 1], color=colors(i), label=f'Cluster {i}')

            cluster_x_coords = [self.cluster_dict[i][0][0] for i in list(self.cluster_dict.keys())]
            cluster_y_coords = [self.cluster_dict[i][0][1] for i in list(self.cluster_dict.keys())]
            ax1.scatter(cluster_x_coords, cluster_y_coords, color='red', marker='x', label='Centroids')

            plt.axis([-15, 15, -15, 15])
            plt.legend()
        elif self.num_features == 3:
            for i in list(self.cluster_dict.keys()):
                cluster_points = np.array(self.cluster_dict[i][1])
                if len(cluster_points) > 0:
                    ax1.scatter(cluster_points[:, 0], cluster_points[:, 1], cluster_points[:, 2], color=colors(i), label=f'Cluster {i}')

            cluster_x_coords = [self.cluster_dict[i][0][0] for i in list(self.cluster_dict.keys())]
            cluster_y_coords = [self.cluster_dict[i][0][1] for i in list(self.cluster_dict.keys())]
            cluster_z_coords = [self.cluster_dict[i][0][2] for i in list(self.cluster_dict.keys())]
            ax1.scatter(cluster_x_coords, cluster_y_coords, cluster_z_coords, color='red', marker='x', label='Centroids')

            ax1.set_xlabel('X')
            ax1.set_ylabel('Y')
            ax1.set_zlabel('Z')
            plt.legend()

        plt.draw()
        plt.pause(0.01)


def main():
    X, y = make_blobs(n_samples=300, centers=4, n_features=3, random_state=counter)
    data = X

    model = KMeans(4)
    print(model.fit(data))


if __name__ == "__main__":
    for i in range(100):
        main()
        counter += 1
