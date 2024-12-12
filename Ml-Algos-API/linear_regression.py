import matplotlib.pyplot as plt
import numpy as np
from numpy import random
import datetime
class LinearRegression:
    def __init__(self, learning_rate=0.02): 
        # Initializing class variables
        self.learning_rate = learning_rate
        # Initializing our parameters to start off as random numbers between 1 and 10
        self.b = 0
        self.params = []

        self.model_refresh = 0.01
        
        self.features_matrix = None
        self.y_labels = None
        
        self.y_preds = None
        self.loss_values = []

    # This function is essentially our model, which is just the equation for a line y = mx + b 
    def f(self, features_matrix, params, b): 
        return np.dot(features_matrix.T, params) + b  # y

    # This is our cost function, which we have chosen to be Mean Squarred Error (Sum of Squarred Error leads to exploding gradients)
    def loss_function(self, features_matrix, y_labels, params, b):
        self.y_preds = self.f(features_matrix, params, b)  # y_hat is a list of our models predictions; produces our y values to plot our line of best fit
        return np.mean((y_labels - self.y_preds) ** 2)  # Returning the MSE for these current parameters

    def fit(self, features_matrix, y_labels):
        # Storing the data passed in as class variables to be used later on
        self.features_matrix = features_matrix  # [[features_1],[features_2], ... [y_labels]]
        self.y_labels = y_labels
        self.params = [random.randint(10) for _ in range(len(features_matrix))]
        # This is our epsilon/small value to be used in calculating our partial derivatives
        h = 1e-5
        # Our model is currently not optimized
        optimized = False

        plt.ion()
        fig = plt.figure(figsize=(12, 8))  # Set figure size
        if len(self.params) == 2:
            ax1 = fig.add_subplot(211, projection='3d')
        else:
            ax1 = fig.add_subplot(211)
        ax2 = fig.add_subplot(212)

        # While its not optimized, run gradient descent
        while not optimized:
            partial_derivatives = []
            for i in range(len(self.params)):  # self.params [w, b, z]
                paramsplush = self.params[:i] + [self.params[i] + h] + self.params[i+1:]
                paramsminush = self.params[:i] + [self.params[i] - h] + self.params[i+1:]
                dparam = (self.loss_function(features_matrix, y_labels, paramsplush, self.b)
                          - self.loss_function(features_matrix, y_labels, paramsminush, self.b)) / (2 * h)
                partial_derivatives.append(dparam)
                self.params[i] = self.params[i] - ((self.learning_rate * dparam))
                print(f"Partial Derivative: {dparam}")
            
            db = (self.loss_function(features_matrix, y_labels, self.params, self.b + h) 
                  - self.loss_function(features_matrix, y_labels, self.params, self.b - h)) / (2 * h)
            self.b = self.b - ((self.learning_rate * db))

            self.y_preds = self.f(self.features_matrix, self.params, self.b)
            current_loss = self.loss_function(features_matrix, y_labels, self.params, self.b)
            self.loss_values.append(current_loss)
            print(f"Current Loss: {current_loss}")

            self.live_plotting(ax1, ax2)

            # If both of these partial derivatives are less than 0.1, we have completed gradient descent
            if all(abs(pd) < 0.01 for pd in partial_derivatives):
                optimized = True

        plt.ioff()
        plt.show()

    def predict(self, features_matrix):
        self.y_preds = self.f(features_matrix, self.params, self.b)
        return self.y_preds

    def plot(self):
        if len(self.params) == 1:
            # Plot our actual x and y points
            plt.plot(self.features_matrix[0], self.y_labels, 'o')
            # Plot our models predictions using the x values
            plt.plot(self.features_matrix[0], self.y_preds)
            plt.show()
        elif len(self.params) == 2:
            fig = plt.figure()
            ax = fig.add_subplot(111, projection='3d')
            ax.scatter(self.features_matrix[0], self.features_matrix[1], self.y_labels, color='red', label='Actual')
            ax.plot_trisurf(self.features_matrix[0], self.features_matrix[1], self.y_preds, color='blue', alpha=0.5, label='Predicted')
            ax.set_xlabel('X Feature')
            ax.set_ylabel('Z Feature')
            ax.set_zlabel('Y Labels')
            plt.show()
        else:
            print(f"Cannot display graph of {len(self.params) + 1} Dimensions")

    def live_plotting(self, ax1, ax2):
        if len(self.params) == 2:
            ax1.cla()
            ax1.scatter(self.features_matrix[0], self.features_matrix[1], self.y_labels, color='red', label='Actual')
            ax1.plot_trisurf(self.features_matrix[0], self.features_matrix[1], self.y_preds, color='blue', alpha=0.5, label='Predicted')
            ax1.set_xlabel('X', fontsize=14)
            ax1.set_ylabel('Z', fontsize=14)
            ax1.set_zlabel('Y', fontsize=14)
        else:
            ax1.cla()
            ax1.plot(self.features_matrix[0], self.y_labels, 'o')
            ax1.plot(self.features_matrix[0], self.y_preds)
            ax1.set_xlabel('Feature', fontsize=14)
            ax1.set_ylabel('Labels', fontsize=14)

        ax2.cla()
        ax2.plot(self.loss_values, label='Loss')
        ax2.set_xlabel('Iteration', fontsize=14)
        ax2.set_ylabel('Loss', fontsize=14)
        ax2.legend()

        plt.draw()
        plt.pause(self.model_refresh)

def main():
    data1 = [
        [0.8, 1.5, 2.3, 3.4, 4.4, 5.2, 5.4, 5.7, 6.2],
        [1.2, 1.5, 2.5, 3.1, 3.3, 3.8, 4.8, 5.5, 6.0],
    ]
    data2 = [
        [0.8, 1.5, 2.3, 3.4, 4.4, 5.2, 5.4, 5.7, 6.2, 6.5, 6.8, 7.0, 7.3, 7.6, 8.0, 8.4, 8.8, 9.0, 9.3, 9.7],  # x_features
        [1.2, 1.0, 2.0, 4.0, 3.5, 6.0, 4.7, 4.4, 7.0, 7.4, 7.8, 8.0, 8.3, 8.6, 9.0, 9.4, 9.8, 10.0, 10.3, 10.7],  # z_features
        [1.2, 1.5, 2.5, 3.1, 3.3, 3.8, 4.8, 5.5, 6.0, 6.3, 6.6, 7.0, 7.3, 7.6, 8.0, 8.3, 8.7, 9.0, 9.4, 9.8],  # y_labels
    ]

    x_features = np.array(data2[0:-1])
    y_label = np.array(data2[-1])

    model = LinearRegression()
    start = datetime.datetime.now()
    model.fit(x_features, y_label)
    end = datetime.datetime.now()
    # model.plot()

if __name__ == "__main__":
    main()
