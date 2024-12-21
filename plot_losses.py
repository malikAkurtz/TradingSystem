import matplotlib.pyplot as plt
import pandas as pd

# File path to the training loss and gradient data
file_path = "training_loss.txt"

# Load the data using pandas
try:
    data = pd.read_csv(file_path)
except FileNotFoundError:
    print(f"Error: File '{file_path}' not found.")
    exit()

# Check if the necessary columns exist
required_columns = {"Epoch", "Loss", "Gradient"}
if not required_columns.issubset(data.columns):
    print(f"Error: File does not contain the required columns: {required_columns}")
    exit()

# Extract the data
epochs = data["Epoch"]
losses = data["Loss"]
gradients = data["Gradient"]

# Plot the data
plt.figure(figsize=(12, 8))

# Plot loss
plt.plot(epochs, losses, marker="o", linestyle="-", color="b", label="Loss")

# Plot gradient magnitudes
plt.plot(
    epochs,
    gradients,
    marker="x",
    linestyle="--",
    color="r",
    label="Gradient Magnitude",
)

# Add titles and labels
plt.title("Training Loss and Gradient Magnitude Over Epochs", fontsize=16)
plt.xlabel("Epoch", fontsize=14)
plt.ylabel("Value", fontsize=14)
plt.grid(True, linestyle="--", alpha=0.6)
plt.legend(fontsize=12)
plt.tight_layout()

# Save the plot
plt.savefig("loss_and_gradient_plot.png")
print("Plot saved as 'loss_and_gradient_plot.png'.")

# Show the plot
plt.show()
