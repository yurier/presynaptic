import numpy as np
import pandas as pd
import os
from scipy.optimize import minimize
import pyqtgraph as pg
from PyQt5.QtWidgets import QApplication, QMainWindow, QWidget, QVBoxLayout
from pyqtgraph import PlotWidget
import time

# Defining the sigmoid function used for vesicle release probability
def sigmoid(z, s , h ):
    return 1 / (1 + np.exp(-s*(z-h)))

# Defining the main simulation function for vesicle release with decay
def vesicle_release_with_decay(D0, R0, tau_adap, delta, tau_D, tau_R, tau_refR, s, h, decay_rate, jump_size, T, dt, release_rate, max_attempts, quiet_duration):
    """
    This function simulates the vesicle release with decay.
    Parameters:
        - D0, R0: Initial values for D (docked) and R (reserve) vesicles.
        - tau_adap, delta: Parameters for Ca_jump adaptation.
        - tau_D, tau_R, tau_refR: Time constants for transitions.
        - s, decay_rate: Parameters for sigmoid function and decay rate.
        - jump_size: Size of the jump in calcium concentration.
        - T, dt: Total time and time step for simulation.
        - release_rate: Rate at which vesicles are released.
        - max_attempts: Maximum number of release attempts.
        - quiet_duration: Duration of quiet period.
    Returns:
        - times, reserve_values, docked_values, Ca_pre_values, Ca_jump_values, Sigmoid_proba, release_times: Simulation results.
    """
    # Initializing vesicle counts, calcium concentration, and Ca_jump
    R, D, Ca_pre, Ca_jump = R0, D0, 0, 1  

    # Initializing lists to store simulation results
    times, reserve_values, docked_values, Ca_pre_values, Ca_jump_values, Sigmoid_proba, release_times = [0], [R], [D], [Ca_pre], [Ca_jump], [0], []

    t, release_attempts, in_quiet_period, quiet_timer = 0, 0, False, 0  

    # Main simulation loop to process each time step
    while t < T + quiet_duration:
        # Compute the time for the next potential vesicle release
        next_release_time = (np.floor(t * release_rate) + 1) / release_rate 

        # Inner loop to process each time step until the next release time or end of simulation
        while t < next_release_time and t < T + quiet_duration: 
            # Handling quiet period
            if in_quiet_period:
                quiet_timer += dt
                if quiet_timer >= quiet_duration:
                    in_quiet_period, quiet_timer = False, 0  

            # Exponential decay of calcium
            Ca_pre *= np.exp(-decay_rate * dt)

            # Update Ca_jump using Euler's method for numerical integration
            dCa_jump = (1 - Ca_jump) * tau_adap - (delta * Ca_jump * Ca_pre)
            Ca_jump += dCa_jump * dt  

            # Random event to determine vesicle movement
            rand_event = np.random.uniform(0, 1)

            # Calculate the rates of different transitions
            transition_RD = ((D0 - D) * R / tau_D) * dt
            transition_DR = ((R0 - R) * D / tau_R) * dt
            replenish_R = ((R0 - R) / tau_refR) * dt

            # Process vesicle movements based on the computed rates
            if rand_event < transition_RD and R > 0: R, D = R - 1, D + 1
            elif rand_event < transition_RD + transition_DR and D > 0: D, R = D - 1, R + 1
            elif rand_event < transition_RD + transition_DR + replenish_R and R < R0: R += 1

            # Updating time and storing simulation results
            t += dt
            times.extend([t])
            reserve_values.extend([R])
            docked_values.extend([D])
            Ca_pre_values.extend([Ca_pre])
            Ca_jump_values.extend([Ca_jump])
            Sigmoid_proba.extend([sigmoid(Ca_pre, s, h)])

        # Attempt to release only if not in a quiet period
        if not in_quiet_period and release_attempts < max_attempts:
            release_attempts += 1
            Ca_pre += jump_size * Ca_jump
            rand = np.random.rand()
            if rand < (sigmoid(Ca_pre, s, h)):
                if D > 0:
                    D -= 1
                    release_times.append(t)

        # Check if we've reached the maximum number of attempts
        if release_attempts == max_attempts:
            in_quiet_period = True

    return times, reserve_values, docked_values, Ca_pre_values, Ca_jump_values, Sigmoid_proba, release_times

# Objective function for optimization with averaging
def objective(params, observed_data, observed_time, jump_size, release_rate, max_attempts, quiet_duration, T, n_iter=10):
    tau_D, tau_R = params  # Only take tau_D and tau_R as parameters
    mse_values = []  # List to store the MSE values for each iteration
    
    for _ in range(100):
        times, reserve_values, docked_values, _, _, _, _ = vesicle_release_with_decay(
            D0, R0, tau_adap, delta, tau_D, tau_R, tau_refR, s, h, decay_rate, jump_size, T, dt, release_rate, max_attempts, quiet_duration)
        simulated_data = (D0 + R0 - np.array(reserve_values) - np.array(docked_values))/(D0 + R0)
        simulated_data_aligned = []
        for t in observed_time:
            closest_time = min(times, key=lambda x: abs(x-t))
            closest_index = times.index(closest_time)
            simulated_data_point = simulated_data[closest_index]
            simulated_data_aligned.append(simulated_data_point)

        mse = np.mean((observed_data - np.array(simulated_data_aligned))**2)
        mse_values.append(mse)  # Append the calculated MSE to the list

    avg_mse = np.mean(mse_values)  # Calculate the average MSE
    mse_values_callback.append(avg_mse)
    return avg_mse



# Initialize the Qt application
app = QApplication([])

# Create main window
main_window = QMainWindow()
central_widget = QWidget()
main_window.setCentralWidget(central_widget)
layout = QVBoxLayout()
central_widget.setLayout(layout)

# Create four PlotWidgets for different plots
plot_widget1 = PlotWidget()
plot_widget2 = PlotWidget()
plot_widget3 = PlotWidget()
plot_widget4 = PlotWidget()
plot_widget5 = PlotWidget()

# Add the PlotWidgets to the layout
layout.addWidget(plot_widget1)
layout.addWidget(plot_widget2)
layout.addWidget(plot_widget3)
layout.addWidget(plot_widget4)
layout.addWidget(plot_widget5)

# Setting the parameters for the simulation
D0 = 25
R0 = 30
tau_D = 5  
tau_R = 45  
tau_refR = 10
s = 0.5
h = 3.0
decay_rate = 2 
jump_size = 1
dt = 0.1
max_attempts = 300
quiet_duration = 100

# Parameter for Ca_jump adaptation
tau_adap = 0.1  
delta = 0.04  

# Load your CSV files
folder_path = "dataset-Fernandez-Alfonso-2008/preprocessed"
csv_files = sorted([os.path.join(folder_path, file) for file in os.listdir(folder_path) if file.endswith('.csv')])
labels = ["2 Hz", "5 Hz", "10 Hz", "20 Hz", "30 Hz"]
frequencies = [int(label.split(' ')[0]) for label in labels]

# Global variable to store MSE values
mse_values_callback = []

# Callback function to execute at each iteration
def callback(params, T, release_rate):
    tau_D, tau_R = params  
    times, reserve_values, docked_values, Ca_pre_values, Ca_jump_values, Sigmoid_proba, release_times = vesicle_release_with_decay(
        D0, R0, tau_adap, delta, tau_D, tau_R, tau_refR, s, h, decay_rate, jump_size, T, dt, release_rate, max_attempts, quiet_duration) 

    simulated_data = (D0 + R0 - np.array(reserve_values) - np.array(docked_values))/(D0 + R0)
    simulated_data_aligned = []
    for t in observed_time:
        closest_time = min(times, key=lambda x: abs(x-t))
        closest_index = times.index(closest_time)
        simulated_data_point = simulated_data[closest_index]
        simulated_data_aligned.append(simulated_data_point)
    
    #mse = np.mean((observed_data - np.array(simulated_data_aligned))**2)  # Calculate MSE here
    #mse_values_callback.append(mse)  # store the mse value

    # Update the plot data in plot_widget1
    plot_widget1.clear()
    plot_widget1.addLegend()
    plot_widget1.plot(observed_time, observed_data, pen='b', name="Observed Data")
    plot_widget1.plot(observed_time, simulated_data_aligned, pen='r', name="Simulated Data")
    
    # Plot vesicle dynamics in plot_widget2
    plot_widget2.clear()
    plot_widget2.addLegend()
    plot_widget2.plot(times, reserve_values, pen='b', name="Reserve Pool (R)")
    plot_widget2.plot(times, docked_values, pen='g', name="Docked Pool (D)")
    plot_widget2.plot(release_times, [D0] * len(release_times), pen=None, symbol='o', symbolBrush='r', name="Release Event")
    
    # Plot calcium dynamics in plot_widget3
    plot_widget3.clear()
    plot_widget3.addLegend()
    plot_widget3.plot(times, Ca_pre_values, pen='c', name="Ca_pre (with jumps and decay)")
    
    # Plot Ca_jump adaptation and Sigmoid probability in plot_widget4
    plot_widget4.clear()
    plot_widget4.addLegend()
    plot_widget4.plot(times, Ca_jump_values, pen='m', name="Ca_jump adaptation")
    plot_widget4.plot(times, Sigmoid_proba, pen='y', name="Probability")
    
    # Plot the MSE in plot_widget5
    plot_widget5.clear()
    plot_widget5.addLegend()
    plot_widget5.plot(mse_values_callback, pen='g', name="MSE")  # Plot the current MSE
    
    # Update the GUI
    time.sleep(0.1)  
    app.processEvents()


# Show the window and run the Qt event loop
main_window.show()

# Main loop for loading each dataset and optimizing parameters
for file, label, freq in zip(csv_files, labels, frequencies):
    T = max_attempts / freq  # Calculate T based on the frequency
    df = pd.read_csv(file, header=None, names=['time', 'deltaF spH'])
    observed_data = df['deltaF spH']
    observed_time = df['time']
    initial_params = [tau_D, tau_R]  # Only pass tau_D and tau_R as initial parameters
    result = minimize(
        objective, 
        initial_params, 
        args=(observed_data, observed_time, jump_size, freq, max_attempts, quiet_duration, T), 
        method='Nelder-Mead',  
        callback=lambda params: callback(params, T, freq),
        options={'xatol': 1e-2, 'fatol': 1e-2, 'disp': True}  # Set tolerance for convergence here
    )
