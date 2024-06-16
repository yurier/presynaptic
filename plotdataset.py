import pandas as pd
import matplotlib.pyplot as plt
import os
from natsort import natsorted

# Path to the directory containing the CSV files
folder_path = "dataset-Fernandez-Alfonso-2008-35C/preprocessed/"

# Get a list of all CSV files in the specified directory
csv_files = natsorted([os.path.join(folder_path, file) for file in os.listdir(folder_path) if file.endswith('.csv')])

# Given labels (assuming there are as many labels as CSV files and they correspond in order)
labels = ["2 Hz", "5 Hz", "10 Hz", "20 Hz", "30 Hz"]

# Initialize a plot
plt.figure(figsize=(12, 8))

# Loop through each file, read its content, and plot on the same figure
for file, label in zip(csv_files, labels):
    # Read CSV without headers and assign column names
    df = pd.read_csv(file, header=None, names=['time', 'deltaF spH'])
    
    plt.plot(df['time'], df['deltaF spH'], label=label)

# Customize the plot
plt.xlabel('Time')
plt.ylabel('mean deltaF spH')
plt.title('Visualization of All Datasets')
plt.legend()
plt.grid(True)
plt.show()
