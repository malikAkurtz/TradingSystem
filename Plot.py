import pandas as pd
import matplotlib.pyplot as plt

def plot_labels_vs_predictions(csv_file):
    # Read the CSV file into a DataFrame
    df = pd.read_csv(csv_file)
    
    # Ensure that columns named 'Label' and 'Prediction' exist
    if 'Label' not in df.columns or 'Prediction' not in df.columns:
        raise ValueError("CSV file must contain 'Label' and 'Prediction' columns.")
    
    # Extract labels and predictions
    labels = df['Label']
    predictions = df['Prediction']
    
    # Plot the data
    plt.figure(figsize=(10, 6))
    plt.scatter(labels, predictions, alpha=0.7, edgecolors='b')
    plt.title("Labels vs. Predictions")
    plt.xlabel("Labels")
    plt.ylabel("Predictions")
    
    # Optionally plot a line y=x for reference
    min_val = min(labels.min(), predictions.min())
    max_val = max(labels.max(), predictions.max())
    plt.plot([min_val, max_val], [min_val, max_val], 'r--', label='Perfect Prediction')
    plt.legend()

    plt.grid(True)
    plt.show()

if __name__ == "__main__":
    # Replace 'results.csv' with the path to your CSV file
    plot_labels_vs_predictions('results.csv')
