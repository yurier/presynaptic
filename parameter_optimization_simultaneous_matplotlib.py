import numpy as np
import pandas as pd
import os
from scipy.optimize import minimize
import matplotlib.pyplot as plt
import time



# Defining the sigmoid function used for vesicle release probability
def sigmoid(z, s , h ):
    return 1 / (1 + np.exp(-s*(z-h)))

def vesicle_release_with_decay(D0, R0, tau_adap, delta, tau_D, tau_R, tau_refR, h, s, decay_rate, jump_size, T_end, dt,  max_attempts, quiet_duration, pre_times):
    """
    This function simulates the vesicle release with decay.
    Parameters:
        - D0, R0: Initial values for D (docked) and R (reserve) vesicles.
        - tau_adap, delta: Parameters for Ca_jump adaptation.
        - tau_D, tau_R, tau_refR: Time constants for transitions.
        - s, decay_rate: Parameters for sigmoid function and decay rate.
        - jump_size: Size of the jump in calcium concentration.
        - T_end, dt: Total time and time step for simulation.
        - max_attempts: Maximum number of release attempts.
        - quiet_duration: Duration of quiet period.
    Returns:
        - times, reserve_values, docked_values, Ca_pre_values, Ca_jump_values, Sigmoid_proba, release_times: Simulation results.
    """
    # Initializing vesicle counts, calcium concentration, and Ca_jump
    R, D, Ca_pre, Ca_jump = R0, D0, 0, 1  

    times, reserve_values, docked_values, Ca_pre_values, Ca_jump_values, Sigmoid_proba, release_times = [0], [R], [D], [Ca_pre], [Ca_jump], [0], []

    t, release_attempts, in_quiet_period, quiet_timer, idx_dt, pre_time_index = 0, 0, False, 0, 0, 0  
    max_iterations = int((T_end + quiet_duration)/dt)

    iteration = 0

    while t < T_end + quiet_duration:
        iteration += 1
        if iteration > max_iterations:
            print("Warning: Exceeded maximum iterations. Breaking loop.")
            break

        # If we've reached the next pre_times
        if pre_time_index < len(pre_times) and t >= pre_times[pre_time_index]:
            pre_time_index += 1

            # Check for release at this exact moment
            if not in_quiet_period and release_attempts <= max_attempts:
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

        # Normal simulation dynamics between pre_times
        if in_quiet_period:
            idx_dt += 1
            quiet_timer = idx_dt*dt
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
        # Gillespie's algorithm-'a-chien (method to simulate the PDMP stochastic part)
        if rand_event < transition_RD and R > 0: R, D = R - 1, D + 1
        elif rand_event < transition_RD + transition_DR and D > 0: D, R = D - 1, R + 1
        elif rand_event < transition_RD + transition_DR + replenish_R and R < R0: R += 1

        # Updating time and storing simulation results
        idx_dt += 1
        t = idx_dt*dt
        times.extend([t])
        reserve_values.extend([R])
        docked_values.extend([D])
        Ca_pre_values.extend([Ca_pre])
        Ca_jump_values.extend([Ca_jump])
        Sigmoid_proba.extend([sigmoid(Ca_pre, s, h)])

    return times, reserve_values, docked_values, Ca_pre_values, Ca_jump_values, Sigmoid_proba,release_times

# Objective function for optimization with averaging for all datasets
def objective_all_datasets(params, all_observed_data, all_observed_time, jump_size, all_frequencies, max_attempts, quiet_duration, n_iter=100):
    tau_D, tau_R, tau_refR, s, h, decay_rate, tau_adap, delta = params
    total_mse = 0  # Variable to store the total MSE for all datasets
    
    # Loop over all datasets
    for observed_data, observed_time, release_rate in zip(all_observed_data, all_observed_time, all_frequencies):
        T_end = (max_attempts/release_rate)  # Total time of simulation
        pre_times = np.linspace(0, max_attempts * (1/release_rate), max_attempts, endpoint=False)

        mse_values = []  # List to store 
        
        for _ in range(n_iter):
            times, reserve_values, docked_values, _, _, _, _ = vesicle_release_with_decay(
                D0, R0, tau_adap, delta, tau_D, tau_R, tau_refR, h, s, decay_rate, jump_size, T_end, dt, max_attempts, quiet_duration, pre_times)
            simulated_data = (D0+R0 - np.array(docked_values) - np.array(reserve_values))/(D0+R0)
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


# Parameters
D0 = 25                               # Initial docked vesicles
R0 = 30                               # Initial reserve vesicles
tau_adap = 5.25686085e-02             # Time constant for calcium adaptation
delta = 2.63467268e-02                # Strength of calcium jump due to AP
tau_D = 3.22059588e+01                # Time constant for vesicle transition from reserve to docked
tau_R = 1.82230449e+01                # Time constant for vesicle transition from docked to reserve
tau_refR = 1.31264672e+01             # Time constant for vesicle replenishment to reserve pool
h = 7.89550730e+00                    # Half-activation calcium concentration for release
s = 3.46225756e-01                    # Steepness of the release sigmoidal relation
decay_rate = 6.48332889e+00           # Rate of calcium decay
jump_size = 1.0                       # Magnitude of calcium jumps
dt = 0.01                             # Time step
release_rate = 30.                    # Probability of release per time step (used for Poisson approximation)
max_attempts = 300                    # Max release attempts before quiet period
quiet_duration = 50.                  # Duration of quiet period

#last optimization [3.22059588e+01 1.82230449e+01 1.31264672e+01 3.46225756e-01 7.89550730e+00 6.48332889e+00 5.25686085e-02 2.63467268e-02]

# Load your CSV files
folder_path = "dataset-Fernandez-Alfonso-2008/preprocessed"
csv_files = sorted([os.path.join(folder_path, file) for file in os.listdir(folder_path) if file.endswith('.csv')])
labels = ["2 Hz", "5 Hz", "10 Hz", "20 Hz", "30 Hz"]
frequencies = [int(label.split(' ')[0]) for label in labels]

# Global variable to store MSE values
mse_values_callback = []

# Callback function to execute at each iteration
def callback(params, all_observed_data, all_observed_time, all_frequencies):
    print(f"Callback for iteration {len(mse_values_callback)}")
    tau_D, tau_R, tau_refR, s, h, decay_rate, tau_adap, delta = params
    avg_freq = sum(all_frequencies) / len(all_frequencies)
    release_rate = avg_freq
    n_rows = 5

    print(params)

    for idx, (observed_data, observed_time, release_rate) in enumerate(zip(all_observed_data, all_observed_time, all_frequencies)):
        print(f"Processing dataset {idx+1}/{len(all_observed_data)} with frequency {release_rate} Hz")
        max_attempts = 300
        T_end = (max_attempts/release_rate)  # Total time of simulation
        pre_times = np.linspace(0, max_attempts * (1/release_rate), max_attempts, endpoint=False)

        times, reserve_values, docked_values, Ca_pre_values, Ca_jump_values, Sigmoid_proba, release_times = vesicle_release_with_decay(
            D0, R0, tau_adap, delta, tau_D, tau_R, tau_refR, h, s, decay_rate, jump_size, T_end, dt,  max_attempts, quiet_duration, pre_times)

        simulated_data = (D0+R0 - np.array(docked_values) - np.array(reserve_values)) / (D0+R0)
        simulated_data_aligned = []
        for t in observed_time:
            closest_time = min(times, key=lambda x: abs(x - t))
            closest_index = times.index(closest_time)
            simulated_data_point = simulated_data[closest_index]
            simulated_data_aligned.append(simulated_data_point)

        for row in range(n_rows):
            ax = axes[row][idx]
            plt.sca(ax)  # Set the current axis to ax
            ax.clear()
            if row == 0:
                # plot original and simulated data
                plt.plot(observed_time, observed_data, 'b', label=f"Observed Data ({release_rate} Hz)")
                plt.plot(observed_time, simulated_data_aligned, 'r', label=f"Simulated Data ({release_rate} Hz)")
                plt.legend()
            elif row == 1:
                # plot reserve, docked and release times
                plt.plot(times, reserve_values, 'g', label="Reserve Values")
                plt.plot(times, docked_values, 'y', label="Docked Values")
                plt.scatter(release_times, [0] * len(release_times), c='r', label="Release Times")
                plt.legend()
                plt.legend()
            elif row == 2:
                # plot Ca_pre_values
                plt.plot(times, Ca_pre_values, 'c', label="Ca_pre Values")
                plt.legend()
            elif row == 3:
                # plot Ca_jump_values and Sigmoid_proba
                plt.plot(times, Ca_jump_values, 'm', label="Ca_jump Values")
                plt.plot(times, Sigmoid_proba, 'w', label="Sigmoid Proba")
                plt.legend()
            elif row == 4:
                # plot mse_values_callback
                plt.plot(range(len(mse_values_callback)), mse_values_callback, 'b', label="MSE Values Callback")
                plt.legend()

    plt.tight_layout()
    plt.draw()
    plt.pause(0.1)

# Initial parameters
initial_params = [tau_D, tau_R, tau_refR, s, h, decay_rate, tau_adap, delta]

# Load all datasets into lists
all_observed_data = []
all_observed_time = []
for file in csv_files:
    df = pd.read_csv(file, header=None, names=['time', 'deltaF spH'])
    all_observed_data.append(df['deltaF spH'])
    all_observed_time.append(df['time'])


plt.figure(figsize=(20, 15))
axes = [[plt.subplot(5, len(frequencies), idx + 1 + row * len(frequencies)) for idx in range(len(frequencies))] for row in range(5)]

# Call the minimize function once, outside of the loop
result = minimize(
    objective_all_datasets,
    initial_params,
    args=(all_observed_data, all_observed_time, jump_size, frequencies, max_attempts, quiet_duration),
    method='Nelder-Mead',
    callback=lambda params: callback(params, all_observed_data, all_observed_time, frequencies),
    options={'disp': True, 'maxiter': 100, 'maxfev': 300}
)

plt.show()
