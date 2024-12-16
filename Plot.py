import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D

def plot_3d_features_vs_prediction_with_plane(csv_file, feature_x="F0", feature_y="F1", prediction_col='Prediction'):
    # Read the CSV into a DataFrame
    df = pd.read_csv(csv_file)

    # Extract the required columns
    X = df[feature_x].values
    Y = df[feature_y].values
    Z = df[prediction_col].values

    # Fit a plane to the data:
    # Prediction ~ c + a*F0 + b*F1
    # We solve for a, b, c using linear least squares.
    # Construct design matrix: [[1, F0, F1], ...]
    A = np.column_stack((np.ones(len(X)), X, Y))
    coeffs, residuals, rank, s = np.linalg.lstsq(A, Z, rcond=None)
    c, a, b = coeffs  # Plane: Z = c + a*X + b*Y

    # Create a grid for plotting the plane
    X_grid = np.linspace(X.min(), X.max(), 50)
    Y_grid = np.linspace(Y.min(), Y.max(), 50)
    X_mesh, Y_mesh = np.meshgrid(X_grid, Y_grid)
    Z_mesh = c + a*X_mesh + b*Y_mesh

    # Plot the data points and the plane
    fig = plt.figure(figsize=(10, 6))
    ax = fig.add_subplot(111, projection='3d')

    # Scatter the original data points
    sc = ax.scatter(X, Y, Z, c=Z, cmap='viridis', alpha=0.8, edgecolors='k')
    cbar = fig.colorbar(sc, ax=ax)
    cbar.set_label('Prediction Value')

    # Plot the fitted plane
    ax.plot_surface(X_mesh, Y_mesh, Z_mesh, alpha=0.3, color='r', edgecolor='none')

    # Set labels
    ax.set_xlabel(feature_x)
    ax.set_ylabel(feature_y)
    ax.set_zlabel(prediction_col)

    plt.title(f"{feature_x} & {feature_y} vs. {prediction_col} with Fitted Plane")
    plt.show()

if __name__ == "__main__":
    # Replace 'results.csv' with your CSV filename if needed
    plot_3d_features_vs_prediction_with_plane("results.csv", feature_x="F2", feature_y="F5", prediction_col='Prediction')
