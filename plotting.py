import pandas as pd
import matplotlib.pyplot as plt

# Load the results.csv file
file_path = "results.csv"  # Update this path if needed
df = pd.read_csv(file_path)

# Extract Labels and Predictions
labels = df["Label"]
predictions = df["Prediction"]

# Plot the labels vs predictions
plt.figure(figsize=(10, 6))
plt.scatter(range(len(labels)), labels, color="blue", label="True Labels", alpha=0.7)
plt.scatter(range(len(predictions)), predictions, color="red", label="Predictions", alpha=0.5)

# Add a line for better visualization
plt.plot(range(len(labels)), labels, linestyle="--", color="gray", alpha=0.3)

# Plot Settings
plt.title("True Labels vs Predictions")
plt.xlabel("Sample Index")
plt.ylabel("Value (0 or 1)")
plt.legend()
plt.grid(True)

# Show the plot
plt.show()
