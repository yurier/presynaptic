import numpy as np
import pandas as pd
import os
from scipy.optimize import minimize
import pyqtgraph as pg
from PyQt5.QtWidgets import QApplication, QMainWindow, QWidget, QVBoxLayout
from pyqtgraph import PlotWidget, plot
import sys  # We need sys so that we can pass argv to QApplication
import time

# Defining the sigmoid function used for vesicle release probability
def sigmoid(z, s=1, h=4):
    return 1 / (1 + np.exp(-s*(z-h)))

# Defining the main simulation function for vesicle release with decay
def vesicle_release_with_decay(D0, R0, tau_D, tau_R, tau_refR, s, decay_rate, jump_size, T, dt, release_rate, max_attempts, quiet_duration):
    
    # Initializing vesicle counts and calcium concentration
    R, D, Ca_pre = R0, D0, 0

    # Initializing lists to store simulation results
    times, reserve_values, docked_values, Ca_pre_values, release_times = [0], [R], [D], [Ca_pre], []

    t, release_attempts, in_quiet_period, quiet_timer = 0, 0, False, 0  # Additional initializations

    # Main simulation loop to process each time step
    while t < T+ quiet_duration:
        # Compute the time for the next potential vesicle release
        next_release_time = (np.floor(t * release_rate) + 1) / release_rate 
        
        # Inner loop to process each time step until the next release time or end of simulation
        while t < next_release_time and t < T + quiet_duration: 
            if in_quiet_period:
                quiet_timer += dt
                if quiet_timer >= quiet_duration:
                    in_quiet_period, quiet_timer = False, 0  # Resetting quiet period and timer

            # Exponential decay of calcium
            Ca_pre *= np.exp(-decay_rate * dt)

            # Random event to determine vesicle movement
            rand_event = np.random.uniform(0, 1)

            # Calculate the rates of different transitions
            transition_RD = (D0 - D) * R / tau_D * dt
            transition_DR = (R0 - R) * D / tau_R * dt
            replenish_R = (R0 - R) / tau_refR * dt

            # Process vesicle movements based on the computed rates
            # Gillespie's algorithm-'a-chien (method to simulate the PDMP stochastic part)
            if rand_event < transition_RD and R > 0: R, D = R - 1, D + 1
            elif rand_event < transition_RD + transition_DR and D > 0: D, R = D - 1, R + 1
            elif rand_event < transition_RD + transition_DR + replenish_R and R < R0: R += 1

            # Updating time and storing simulation results
            t += dt
            times.extend([t])
            reserve_values.extend([R])
            docked_values.extend([D])
            Ca_pre_values.extend([Ca_pre])
            
        # Attempt to release only if not in a quiet period
        if not in_quiet_period and release_attempts < max_attempts:
            release_attempts += 1
            Ca_pre += jump_size
            if np.random.rand() < sigmoid(Ca_pre, s):
                if D > 0:
                    D -= 1
                    release_times.append(t)

        # Check if we've reached the maximum number of attempts
        if release_attempts == max_attempts:
            in_quiet_period = True


    return times, reserve_values, docked_values, Ca_pre_values, release_times

def objective(params, observed_data, observed_time, jump_size, release_rate, max_attempts, quiet_duration):
    D0, R0, tau_D, tau_R, tau_refR, s, decay_rate = params
    times, reserve_values, docked_values, _, _ = vesicle_release_with_decay(
        D0, R0, tau_D, tau_R, tau_refR, s, decay_rate, jump_size, T, dt, release_rate, max_attempts, quiet_duration)
    simulated_data = D0 + R0 - np.array(reserve_values) - np.array(docked_values) 
    simulated_data_aligned = []
    for t in observed_time:
        closest_time = min(times, key=lambda x: abs(x-t))
        closest_index = times.index(closest_time)
        simulated_data_point = simulated_data[closest_index]
        simulated_data_aligned.append(simulated_data_point)
        
    mse = np.mean((observed_data - np.array(simulated_data_aligned))**2)
    return mse


# Initialize the Qt application
app = QApplication([])

# Create main window
main_window = QMainWindow()
central_widget = QWidget()
main_window.setCentralWidget(central_widget)
layout = QVBoxLayout()
central_widget.setLayout(layout)
plot_widget = PlotWidget()
layout.addWidget(plot_widget)

# Setting the parameters for the simulation
D0 = 25
R0 = 30
tau_D = 10.0  
tau_R = 80.0  
tau_refR = 20.0  
s = 2.0
decay_rate = 1.2  
jump_size = 1
T = 20
dt = 0.01
release_rate = 5.0
max_attempts=100
quiet_duration=100

# Load your CSV files
folder_path = "dataset-Fernandez-Alfonso-2008/preprocessed"

csv_files = sorted([os.path.join(folder_path, file) for file in os.listdir(folder_path) if file.endswith('.csv')])
labels = ["2 Hz", "5 Hz", "10 Hz", "20 Hz", "30 Hz"]
frequencies = [int(label.split(' ')[0]) for label in labels]

# Callback function to execute at each iteration
def callback(params):
    D0, R0, tau_D, tau_R, tau_refR, s, decay_rate = params
    # Run the simulation
    times, reserve_values, docked_values, _, _ = vesicle_release_with_decay(
        D0, R0, tau_D, tau_R, tau_refR, s, decay_rate, jump_size, T, dt, release_rate, max_attempts, quiet_duration)
    # Compute the simulated data
    simulated_data = (D0 + R0 - np.array(reserve_values) - np.array(docked_values))/(D0 + R0)
    # Align the simulated data with the observed time points
    simulated_data_aligned = []
    for t in observed_time:
        closest_time = min(times, key=lambda x: abs(x-t))
        closest_index = times.index(closest_time)
        simulated_data_point = simulated_data[closest_index]
        simulated_data_aligned.append(simulated_data_point)
    # Update the plot data
    plot_widget.plot(observed_time, observed_data, pen='b', clear=True)
    plot_widget.plot(observed_time, simulated_data_aligned, pen='r')
    # Update the GUI
    time.sleep(0.1)  # add a small delay to allow the plot to update
    app.processEvents()


# Show the window and run the Qt event loop
main_window.show()

# Main loop for loading each dataset and optimizing parameters
for file, label, freq in zip(csv_files, labels, frequencies):
    df = pd.read_csv(file, header=None, names=['time', 'deltaF spH'])
    observed_data = df['deltaF spH']
    observed_time = df['time']
    release_rate = freq
    initial_params = [D0, R0, tau_D, tau_R, tau_refR, s, decay_rate]
    result = minimize(objective, initial_params, args=(observed_data, observed_time, jump_size, release_rate, max_attempts, quiet_duration), method='Nelder-Mead', callback=callback)
