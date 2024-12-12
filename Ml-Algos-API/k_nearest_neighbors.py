
from sklearn.datasets import make_blobs
import numpy as np
import matplotlib.pyplot as plt
class KNearestNeighbors:
    def __init__(self):
        self.k = None
        self.data_point = None

    
    def classify_Point(self, features,labels, data_point, k):
        self.k = k
        self.data_point = data_point
        master_dict = {} #{point : (distance, cluster)}
        
        for i in range(len(features)):           # [x,y,z]
            distance = np.linalg.norm(data_point - features[i])
            master_dict[i] = (distance, features[i], labels[i])
            #print(master_dict[i])

        master_dict = dict(sorted(master_dict.items(), key=lambda x: x[1][0]))
        
        
        cluster_dict = {}
        for i in range(len(set(labels))):
            cluster_dict[i] = 0
        
        first_k_values = list(master_dict.values())[:self.k]

        for value in first_k_values:
            cluster_dict[value[2]] = cluster_dict[value[2]] + 1

        print(cluster_dict)
        cluster_tuple = sorted(cluster_dict.items(), key=lambda x: x[1])
        print(cluster_tuple)

        return cluster_tuple[-1][0]





def main():

    X, y = make_blobs(n_samples=300, centers=4, n_features=8, random_state=1)
    features = X
    labels = y
    
    

    fig = plt.figure(figsize=(12, 8))
    # ax1 = fig.add_subplot(111, projection='3d')
    ax1 = fig.add_subplot(111)
    #ax1.scatter(features[:, 0], features[:, 1], features[:, 2], c = labels, cmap='viridis')
    ax1.scatter(features[:, 0], features[:, 1], c = labels, cmap='viridis')
    ax1.scatter(-5, -6, c = 'black')
    ax1.set_xlabel('X')
    ax1.set_ylabel('Y')
    #ax1.set_zlabel('Z')
    plt.show()

    model = KNearestNeighbors()
    print(model.classify_Point(features, labels, [-5,-6,4,6,7,4,3,8], 10))
    

    
    
    

if __name__ == "__main__":
    main()
        