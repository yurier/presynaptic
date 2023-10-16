import numpy as np
import pandas as pd
import os
from scipy.optimize import minimize
import pyqtgraph as pg
from PyQt5.QtWidgets import QApplication, QMainWindow, QWidget, QVBoxLayout
from pyqtgraph import PlotWidget
import time
from PyQt5.QtWidgets import QGridLayout

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

# Objective function for optimization with averaging for all datasets
def objective_all_datasets(params, all_observed_data, all_observed_time, jump_size, all_frequencies, max_attempts, quiet_duration, n_iter=100):
    tau_D, tau_R, tau_refR, s, h, decay_rate, tau_adap, delta = params
    total_mse = 0  # Variable to store the total MSE for all datasets
    
    # Loop over all datasets
    for observed_data, observed_time, freq in zip(all_observed_data, all_observed_time, all_frequencies):
        T = max_attempts / freq  # Calculate T based on the frequency
        mse_values = []  # List to store 
        
        for _ in range(n_iter):
            times, reserve_values, docked_values, _, _, _, _ = vesicle_release_with_decay(
                D0, R0, tau_adap, delta, tau_D, tau_R, tau_refR, s, h, decay_rate, jump_size, T, dt, freq, max_attempts, quiet_duration)
            simulated_data = (D0 - np.array(docked_values))/(D0)
            simulated_data_aligned = []
            for t in observed_time:
                closest_time = min(times, key=lambda x: abs(x-t))
                closest_index = times.index(closest_time)
                simulated_data_point = simulated_data[closest_index]
                simulated_data_aligned.append(simulated_data_point)

            mse = np.mean((observed_data - np.array(simulated_data_aligned))**2)
            mse_values.append(mse)  # Append the calculated MSE to the list

        avg_mse = np.mean(mse_values)  # Calculate the average MSE for the current dataset
        total_mse += avg_mse  # Add the average MSE to the total MSE
    
    mse_values_callback.append(total_mse)
    return total_mse


# Setting the parameters for the simulation
D0 = 25
R0 = 30
tau_D = 2.68104323e+01#2.60381944e+01#2.51406722e+01  
tau_R = 2.02212386e+01#2.01812934e+01#1.97873162e+01  
tau_refR = 2.24039034e+01#2.22210394e+01#2.17659560e+01
s = 7.96026828e-01#7.90609246e-01#0.8
h = 5.26144961e+00#5.26474562e+00#6.08386430e+00
decay_rate = 4
jump_size = 1
dt = 0.05
max_attempts = 300
quiet_duration = 50

# Parameter for Ca_jump adaptation
tau_adap = 5.24849781e-02#5.24280854e-02#5.12249414e-02 
delta = 2.09973392e-02#2.09796610e-02#1.98393158e-02

# Load your CSV files
folder_path = "dataset-Fernandez-Alfonso-2008/preprocessed"
csv_files = sorted([os.path.join(folder_path, file) for file in os.listdir(folder_path) if file.endswith('.csv')])
labels = ["2 Hz", "5 Hz", "10 Hz", "20 Hz", "30 Hz"]
frequencies = [int(label.split(' ')[0]) for label in labels]


# Initialize the Qt application
app = QApplication([])

# Create main window
main_window = QMainWindow()
central_widget = QWidget()
main_window.setCentralWidget(central_widget)
layout = QGridLayout()  # Use QGridLayout instead of QVBoxLayout
central_widget.setLayout(layout)

# Calculate the grid size based on the number of datasets
n_datasets = len(csv_files)
grid_size = n_datasets

# Create a grid of PlotWidgets for different plots
plot_widgets = [[PlotWidget() for _ in range(grid_size)] for _ in range(grid_size)]

# Add the PlotWidgets to the layout
for i in range(grid_size):
    for j in range(grid_size):
        layout.addWidget(plot_widgets[i][j], i, j)

# Global variable to store MSE values
mse_values_callback = []

# Callback function to execute at each iteration
def callback(params, all_observed_data, all_observed_time, all_frequencies):
    tau_D, tau_R, tau_refR, s, h, decay_rate, tau_adap, delta = params
    avg_freq = sum(all_frequencies) / len(all_frequencies)
    release_rate = avg_freq
    n_rows=5
    print(params)
    for idx, (observed_data, observed_time, freq) in enumerate(zip(all_observed_data, all_observed_time, all_frequencies)):
        
        T = 300 / freq
        times, reserve_values, docked_values, Ca_pre_values, Ca_jump_values, Sigmoid_proba, release_times = vesicle_release_with_decay(
            D0, R0, tau_adap, delta, tau_D, tau_R, tau_refR, s, h, decay_rate, jump_size, T, dt, freq, max_attempts, quiet_duration)

        simulated_data = (D0 - np.array(docked_values)) / (D0)
        simulated_data_aligned = []
        for t in observed_time:
            closest_time = min(times, key=lambda x: abs(x - t))
            closest_index = times.index(closest_time)
            simulated_data_point = simulated_data[closest_index]
            simulated_data_aligned.append(simulated_data_point)

        for row in range(n_rows):
            plot_widget = plot_widgets[row][idx]
            plot_widget.clear()
            plot_widget.addLegend()

            if row == 0:
                # plot original and simulated data
                plot_widget.plot(observed_time, observed_data, pen='b', name=f"Observed Data ({freq} Hz)")
                plot_widget.plot(observed_time, simulated_data_aligned, pen='r', name=f"Simulated Data ({freq} Hz)")
            elif row == 1:
                # plot reserve, docked and release times
                plot_widget.plot(times, reserve_values, pen='g', name="Reserve Values")
                plot_widget.plot(times, docked_values, pen='y', name="Docked Values")
                plot_widget.plot(release_times, [0] * len(release_times), pen=None, symbol='o', symbolBrush='r', name="Release Times")
            elif row == 2:
                # plot Ca_pre_values
                plot_widget.plot(times, Ca_pre_values, pen='c', name="Ca_pre Values")
            elif row == 3:
                # plot Ca_jump_values and Sigmoid_proba
                plot_widget.plot(times, Ca_jump_values, pen='m', name="Ca_jump Values")
                plot_widget.plot(times, Sigmoid_proba, pen='w', name="Sigmoid Proba")
            elif row == 4:
                # plot mse_values_callback
                plot_widget.plot(list(range(len(mse_values_callback))), mse_values_callback, pen='b', name="MSE Values Callback")

    time.sleep(0.1)
    app.processEvents()



# Show the window and run the Qt event loop
main_window.show()
# Initial parameters
initial_params = [tau_D, tau_R, tau_refR, s, h, decay_rate, tau_adap, delta]

# Load all datasets into lists
all_observed_data = []
all_observed_time = []
for file in csv_files:
    df = pd.read_csv(file, header=None, names=['time', 'deltaF spH'])
    all_observed_data.append(df['deltaF spH'])
    all_observed_time.append(df['time'])

# Call the minimize function once, outside of the loop
result = minimize(
    objective_all_datasets,
    initial_params,
    args=(all_observed_data, all_observed_time, jump_size, frequencies, max_attempts, quiet_duration),
    method='Nelder-Mead',
    callback=lambda params: callback(params, all_observed_data, all_observed_time, frequencies),
    options={'disp': True, 'maxiter': 100, 'maxfev': 300}
)