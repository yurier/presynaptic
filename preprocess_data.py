import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os

def preprocess_data(df, desired_time_step=0.1):
    """
    Preprocess the given DataFrame.
    
    Parameters:
    - df: Input DataFrame with 'time' and 'deltaF spH' columns
    - desired_time_step: The consistent time step to resample the data
    
    Returns:
    - Processed DataFrame
    """
    # Sort the data by 'time' column
    df = df.sort_values(by='time').reset_index(drop=True)
    
    # If the dataset does not have a value at time 0, interpolate one
    if 0 not in df['time'].values:
        # Find the data point that's immediately before 0
        before_zero = df[df['time'] < 0].iloc[-1] if not df[df['time'] < 0].empty else None
        after_zero = df[df['time'] > 0].iloc[0]
        
        if before_zero is not None:
            slope = (after_zero['deltaF spH'] - before_zero['deltaF spH']) / (after_zero['time'] - before_zero['time'])
            intercept = after_zero['deltaF spH'] - slope * after_zero['time']
            
            value_at_zero = slope * 0 + intercept
            new_row = pd.DataFrame({'time': [0], 'deltaF spH': [value_at_zero]})
            df = pd.concat([df, new_row]).sort_values(by='time').reset_index(drop=True)
    
    # Define the start and end times explicitly
    start_time = 0
    end_time = df['time'].max()
    
    # Resample data to a consistent time step using linear interpolation
    new_times = np.arange(start_time, end_time, desired_time_step)
    interpolated_values = np.interp(new_times, df['time'], df['deltaF spH'], left=np.nan, right=np.nan)
    
    # Create a new DataFrame from interpolated values
    new_df = pd.DataFrame({'time': new_times, 'deltaF spH': interpolated_values})
    
    # Handle missing values (just in case, after interpolation)
    new_df['deltaF spH'].interpolate(method='linear', inplace=True)
    
    return new_df



# Normalize data (Optional)
def normalize_data(df):
    """
    Normalize the 'deltaF spH' column to have values between 0 and 1.
    
    Parameters:
    - df: Input DataFrame with 'time' and 'deltaF spH' columns
    
    Returns:
    - Normalized DataFrame
    """
    min_val = df['deltaF spH'].min()
    max_val = df['deltaF spH'].max()
    
    df['deltaF spH'] = (df['deltaF spH'] - min_val) / (max_val - min_val)
    
    return df


# Path to the directory containing the CSV files
folder_path = "dataset-Fernandez-Alfonso-2008"

# Get a list of all CSV files in the specified directory
csv_files = sorted([os.path.join(folder_path, file) for file in os.listdir(folder_path) if file.endswith('.csv')])

# Create a directory to store the preprocessed data
output_directory = os.path.join(folder_path, "preprocessed")
if not os.path.exists(output_directory):
    os.makedirs(output_directory)


# Process and optionally normalize each dataset
preprocessed_data = []
original_data = []

for file in csv_files:
    df = pd.read_csv(file, header=None, names=['time', 'deltaF spH'])
    original_data.append(df.copy())  # Store original data for comparison
    df = preprocess_data(df)
    # Uncomment if normalization is desired
    # df = normalize_data(df)
    preprocessed_data.append(df)
    
    # Save preprocessed data to a new CSV file
    base_name = os.path.basename(file)  # Get original filename
    output_path = os.path.join(output_directory, base_name)
    df.to_csv(output_path, index=False, header=False)

# Visualization of Original vs. Preprocessed Data
num_files = len(csv_files)

plt.figure(figsize=(15, 5 * num_files))

for i, (original_df, preprocessed_df, label) in enumerate(zip(original_data, preprocessed_data, labels)):
    # Original Data
    plt.subplot(num_files, 2, i*2 + 1)
    plt.plot(original_df['time'], original_df['deltaF spH'], label=f"Original {label}")
    plt.xlabel('Time')
    plt.ylabel('mean deltaF spH')
    plt.title(f'Original Data ({label})')
    plt.grid(True)
    plt.legend()
    
    # Preprocessed Data
    plt.subplot(num_files, 2, i*2 + 2)
    plt.plot(preprocessed_df['time'], preprocessed_df['deltaF spH'], label=f"Preprocessed {label}", color='orange')
    plt.xlabel('Time')
    plt.ylabel('mean deltaF spH (preprocessed)')
    plt.title(f'Preprocessed Data ({label})')
    plt.grid(True)
    plt.legend()

plt.tight_layout()
plt.show()