import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs
import pandas as pd
import time as tm

counter = 0
class KMeans():
    def __init__(self, k):
        # Initializing our # of clusters to be the k value passed in upon model creation
        self.num_clusters = k
        # This dictionary's keys will be the cluster #, and its value will be a list of format
        # [centroid coordinates in form [x, y, z...], [list of coordinates of all other points related to this centroid]]
        self.cluster_dict = {}
        # Where our classes data will be stored
        self.data = None
        # Variable to represent the number of features/the dimensionality of the data
        self.num_features = None
        # Variable representing if we are dealing with a dataframe
        self.df = False
        self.num_iterations = 1

    # Function to apply the K-Means Clustering
    def fit(self, data_coords):
        # If it is a dataframe
        if isinstance(data_coords, pd.DataFrame):
            # Create a dataframe from the data
            data_coords_df = data_coords.copy()
            # Set data_coords (which is currently a dataframe) to be the values of that dataframe as a Numpy array
            data_coords = data_coords.values
            # Create a new column in our dataframe representation of the data to hold the cluster of the point
            data_coords_df["cluster"] = None
            # We are dealing with a dataframe
            self.df = True
        # If it is not a dataframe, make it one
        else:
            # Create a dataframe out of the Numpy array that was passed into this function
            data_coords_df = pd.DataFrame(data_coords)
            # Again create a cluster column to hold the cluster of each point
            data_coords_df["cluster"] = None
            # We are now dealing with a dataframe
            self.df = True

        # Setting the classes data to be the Numpy array of data_coords
        self.data = data_coords
        # The number of features in our data will be specified by the second index of the shape tuple
        self.num_features = self.data.shape[1]
        # This dictionary will store, as keys, the iteration of k means, and as the value, a list with index 0
        # being the variance of the clusters for this iteration, and index 1 being the clusters array which is returned
        variance_clusters_dict = {}
        
        # Plotting stuff
        plt.ion()
        fig = plt.figure(figsize=(12, 8))
        if self.num_features == 2:
            ax1 = fig.add_subplot(111)
        elif self.num_features == 3:
            ax1 = fig.add_subplot(111, projection='3d')

        # The number of iterations that we will run K Means with randomly initializing centroids to see which gives
        # the lowest variance and eventually returning the cluster that result in the lowest variance
        for m in range(self.num_iterations):
            # Refreshing our cluster dictionary at the start of every iteration
            self.cluster_dict = {}
            # For every cluster
            for cluster in range(self.num_clusters):
                # Initialize a list that will hold the coordinates of its centroid
                cluster_coordinates = []
                # For each one of our features
                for i in range(self.num_features):
                    # Calculate the mean of all the features in that feature column
                    feature_column_mean = np.mean(self.data[:, i])
                    # Add a little bit of noise
                    random_noise = np.random.normal(0, 0.1)
                    # Append this value to our coordinates list, this will be the coordinate of our centroid for that
                    # feature/dimension
                    cluster_coordinates.append(feature_column_mean + random_noise)
                # Once we have created our coordinates for our cluster, set the value of the dictionary at index 1
                # to be list list of coordinates and set the points belonging to this centroid to an empty list
                self.cluster_dict[cluster] = [np.array(cluster_coordinates), []]

            # Loop to apply K Means Clustering
            while True:
                # If the dimensionality of the data is less than 3, we can plot it
                if self.num_features <= 3:
                    self.live_plotting(ax1)
                # For every cluster
                for i in list(self.cluster_dict.keys()):
                    # Refresh the points belonging to this cluster centroid
                    self.cluster_dict[i][1] = []
                # For every point in data coordinates
                for j, data_coord in enumerate(data_coords):
                    # Initialize the distance to the closest clusters to infinity
                    closest_cluster_distance = float('inf')
                    # The closest cluster will be set to nothing
                    closest_cluster = None
                    # For every cluster
                    for i in list(self.cluster_dict.keys()):
                        # Store the cluster's centroid coordinates
                        cluster_coord = self.cluster_dict[i][0]
                        # Calculate the distance from the cluster to the datapoint
                        distance = np.linalg.norm(cluster_coord - data_coord)
                        # If this distance is less that the closest cluster's distance,
                        # it is now the new closest cluster
                        if distance < closest_cluster_distance:
                            closest_cluster_distance = distance
                            closest_cluster = i
                    # This point now belongs to this centroid, so add it to the centroids points list
                    self.cluster_dict[closest_cluster][1].append(data_coord)
                    # If we are dealing with a dataframe
                    if self.df:
                        # Set the cluster column at that point to be this new closest cluster
                        data_coords_df.at[j, "cluster"] = closest_cluster

                # Assume our clusters have converged
                converged = True
                # For every cluster
                for i in list(self.cluster_dict.keys()):
                    # Store the centroids current coordinates before it is updated
                    old_centroid = np.array(self.cluster_dict[i][0])
                    # Store the # of points belonging to this centroid
                    num_points_in_cluster = len(self.cluster_dict[i][1])
                    # If the centroid has at least one point belonging to it
                    if (num_points_in_cluster) > 0:
                        # Set the centroid to be the average of all the points now belonging to it
                        self.cluster_dict[i][0] = np.mean(self.cluster_dict[i][1], axis=0)
                    # Store this new value that we just calculated as the new centroids coordinates
                    new_centroid = np.array(self.cluster_dict[i][0])
                    # Calculate the change from the previous centroids coordinates to its new recalculated coordinates
                    mean_change = np.linalg.norm(new_centroid - old_centroid)
                    # If the change was greater than 0.01, we are actually not yet converged, set it back to False
                    if mean_change > 0.01:
                        converged = False
                # If we get through every cluster and the mean_change is less than 0.01 then we are converged, break
                if converged:
                    break
            # Now that this iteration is over, we will calculate the variance that is produced from these centroids
            # Calculate the sum of each of the variances of the centroids and their points (we want this to be low)
            variance_sum_for_iteration = np.sum(list(self.calculate_cluster_variance().values()))
            # Append this value to a dictionary that will store the {iteration : its variance, and the cluster assignments]}
            variance_clusters_dict[m] = [variance_sum_for_iteration, data_coords_df['cluster'].values.copy()]
        
        print("DONE")
        print(variance_clusters_dict)
        
        # We will now determine which iteration produced the lowest variance in order to return that list of cluster
        # assignments
        # At first our best cluster is None and our lowest_variance is infinity
        best_iteration = None
        lowest_variance = float('inf')
        # For each iteration
        for iteration in variance_clusters_dict.keys():
            # We will check if this iterations variance is lower than our current lowest variance
            if variance_clusters_dict[iteration][0] < lowest_variance:
                # If it is then set the best iteration to this iteration
                best_iteration = iteration
                # And the lowest variance to the variance stored in the dictionary at this index
                lowest_variance = variance_clusters_dict[iteration][0]

        # Return the cluster assignments belonging to the best iteration
        return variance_clusters_dict[best_iteration][1]

    # Function for calculating variance among clusters
    def calculate_cluster_variance(self):
        # Dictionary to store each cluster and its variance
        variance_dict = {}
        # For each cluster
        for cluster in range(self.num_clusters):
            # Store the coordinates of the points within that cluster
            cluster_points = np.array(self.cluster_dict[cluster][1])
            # Store te coordinates of the centroid
            centroid = self.cluster_dict[cluster][0]
            # If there are points belonging to the cluster
            if len(cluster_points) > 0:
                # Calculate the variance for that cluster and store it in the dictionary
                variance_dict[cluster] = np.mean(np.linalg.norm(cluster_points - centroid, axis=1) ** 2)
            else:
                # If there are not points belonging to the centroid, the variance is 0
                variance_dict[cluster] = 0
        # Return the dictionary
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
        plt.pause(0.1)
        ax1.cla()


def main():
    X, y = make_blobs(n_samples=300, centers=4, n_features=3, random_state=counter)
    data = X

    model = KMeans(4)
    print(model.fit(data))


if __name__ == "__main__":
    for i in range(100):
        main()
        counter += 1
