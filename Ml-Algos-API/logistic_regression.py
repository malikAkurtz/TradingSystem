import matplotlib.pyplot as plt
import numpy as np
from numpy import random

class LogisticRegression:
    def __init__(self, learning_rate=0.005):
        self.learning_rate = learning_rate
        self.b = 0
        self.params = []
        self.model_refresh = 0.01
        self.features_matrix = None
        self.y_labels = None
        self.floating_y_preds = None
        self.y_preds = None
        self.loss_values = []  # Initialize loss values list

    def f(self, features_matrix, params, b):
        return np.dot(features_matrix.T, params) + b

    def cast_to_p(self, unscaled_pred):
        num_to_e = np.exp(unscaled_pred)
        p = (num_to_e) / (num_to_e + 1)
        return p

    def loss_function(self, features_matrix, y_labels, params, b):
        self.floating_y_preds = np.array([self.cast_to_p(unscaled_p) for unscaled_p in self.f(features_matrix, params, b)])
        return np.sum(((-1 * y_labels) * np.log(self.floating_y_preds)) - ((1 - y_labels) * np.log(1.0001 - self.floating_y_preds)))

    def fit(self, features_matrix, y_labels):
        self.features_matrix = features_matrix
        self.y_labels = y_labels
        self.params = [random.randint(10) for _ in range(len(features_matrix))]
        h = 1e-5
        optimized = False

        plt.ion()
        fig = plt.figure(figsize=(12,8))
        if len(self.params) == 2:
            ax1 = fig.add_subplot(211, projection='3d')
        if len(self.params) == 1:
            ax1 = fig.add_subplot(211)
        ax2 = fig.add_subplot(212)

        while not optimized:
            partial_derivatives = []
            for i in range(len(self.params)):
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

            self.y_preds = np.heaviside(self.f(self.features_matrix, self.params, self.b), 0.5)
            current_loss = self.loss_function(features_matrix, y_labels, self.params, self.b)
            self.loss_values.append(current_loss)
            print(f"Current Loss: {current_loss}")

            self.live_plotting(ax1, ax2)

            if all(abs(pd) < 0.01 for pd in partial_derivatives):
                optimized = True

        plt.ioff()
        plt.show()

    def predict(self, features_matrix):
        self.y_preds = np.heaviside(self.f(features_matrix, self.params, self.b), 0.5)
        return self.y_preds

    def plot(self):
        if len(self.params) == 1:
            plt.plot(self.features_matrix[0], self.y_labels, 'o')
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
            ax1.set_xlabel('X')
            ax1.set_ylabel('Z')
            ax1.set_zlabel('Y')
        if len(self.params) == 1:
            ax1.cla()
            ax1.plot(self.features_matrix[0], self.y_labels, 'o')
            ax1.plot(self.features_matrix[0], self.y_preds)

        ax2.cla()
        ax2.plot(self.loss_values, label='Loss')
        ax2.set_xlabel('Iteration')
        ax2.set_ylabel('Loss')
        ax2.legend()

        plt.draw()
        plt.pause(self.model_refresh)


def main():
    data = [
        [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19],  # Weight of a mouse in pounds
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 1, 1, 1, 1, 1, 1],  # 0 = Not obese, 1 = Obese
    ]

    x_features = np.array(data[0:-1])
    y_label = np.array(data[-1])

    model = LogisticRegression()
    model.fit(x_features, y_label)
    print(model.predict(x_features))


if __name__ == "__main__":
    main()
