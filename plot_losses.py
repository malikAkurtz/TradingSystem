import matplotlib.pyplot as plt
import pandas as pd

# File path to the training loss file
file_path = "training_loss.txt"

# Load the data using pandas
try:
    data = pd.read_csv(file_path)
except FileNotFoundError:
    print(f"Error: File '{file_path}' not found.")
    exit()

# Check if the necessary columns exist
if "Epoch" not in data.columns or "Loss" not in data.columns:
    print("Error: File does not contain 'Epoch' or 'Loss' columns.")
    exit()

# Extract the data
epochs = data["Epoch"]
losses = data["Loss"]

# Plot the data
plt.figure(figsize=(10, 6))
plt.plot(epochs, losses, marker="o", linestyle="-", color="b", label="Loss")
plt.title("Training Loss Over Epochs", fontsize=16)
plt.xlabel("Epoch", fontsize=14)
plt.ylabel("Loss", fontsize=14)
plt.grid(True, linestyle="--", alpha=0.6)
plt.legend(fontsize=12)
plt.tight_layout()

# Save the plot
plt.savefig("training_loss_plot.png")
print("Plot saved as 'training_loss_plot.png'.")

# Show the plot
plt.show()
