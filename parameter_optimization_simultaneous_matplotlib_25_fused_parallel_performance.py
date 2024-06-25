import numpy as np
import pandas as pd
import os
from scipy.optimize import minimize
import matplotlib.pyplot as plt
import time
from natsort import natsorted
from concurrent.futures import ThreadPoolExecutor
import pyautogui
from scipy.optimize import Bounds

# Defining the sigmoid function used for vesicle release probability
def sigmoid(Ca_pre, s):
    return 1 / (1 + np.exp(-s*(Ca_pre-8)))



def vesicle_release_with_decay(D0, R0, F0, tau_adap, delta, tau_D, tau_R, tau_refR, s, decay_rate, jump_size, T_end, dt, max_attempts, quiet_duration, pre_times):
    """
    This function simulates the vesicle release with decay.
    Parameters:
        - D0, R0, F0: Initial values for D (docked), R (reserve) vesicles and F (fused).
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
    R, D, F, Ca_pre, Ca_jump = int(R0), int(D0), F0, 0, 1  

    max_iterations = int((T_end + quiet_duration) / dt) + 1  # +1 to account for t=0

    # Preallocating arrays
    times = np.zeros(max_iterations)
    reserve_values = np.zeros(max_iterations)
    docked_values = np.zeros(max_iterations)
    fused_values = np.zeros(max_iterations)
    Ca_pre_values = np.zeros(max_iterations)
    Ca_jump_values = np.zeros(max_iterations)
    Sigmoid_proba = np.zeros(max_iterations)
    release_times = []
    spike_times = []

    # Initial values
    times[0] = 0
    reserve_values[0] = R
    docked_values[0] = D
    fused_values[0] = F
    Ca_pre_values[0] = Ca_pre
    Ca_jump_values[0] = Ca_jump
    Sigmoid_proba[0] = sigmoid(Ca_pre, s)

    t, release_attempts, in_quiet_period, quiet_timer, idx_dt, pre_time_index = 0, 0, False, 0, 0, 0  

    iteration = 0

    while t < T_end + quiet_duration:
        iteration += 1
        if iteration >= max_iterations:
            print("Warning: Exceeded maximum iterations. Breaking loop.")
            break

        # If we've reached the next pre_time
        if pre_time_index < len(pre_times) and t >= pre_times[pre_time_index]:
            pre_time_index += 1
            # Check for release at this exact moment
            if not in_quiet_period and release_attempts <= max_attempts:
                release_attempts += 1
                Ca_pre += jump_size * Ca_jump
                rand = np.random.rand()
                spike_times.append(t)
                if rand < sigmoid(Ca_pre, s):  # Use previously calculated probability
                    if D > 0:
                        D -= 1
                        F += 1
                        release_times.append(t)

            # Check if we've reached the maximum number of attempts
            if release_attempts == max_attempts:
                in_quiet_period = True

        # Normal simulation dynamics between pre_times
        if in_quiet_period:
            idx_dt += 1
            quiet_timer = idx_dt * dt
            if quiet_timer >= quiet_duration:
                in_quiet_period, quiet_timer = False, 0  

        # Exponential decay of calcium
        Ca_pre *= np.exp(- dt/decay_rate)

        # Update Ca_jump using Euler's method for numerical integration
        dCa_jump = ((1 - Ca_jump) / tau_adap) - (delta * Ca_jump * Ca_pre)
        Ca_jump += dCa_jump * dt

        # Random event to determine vesicle movement
        rand_event = np.random.uniform(0, 1)

        # Calculate the rates of different transitions
        transition_RD = ((D0 - D) * R / tau_D) * dt
        transition_DR = ((R0 - R) * D / tau_R) * dt
        replenish_R = ((R0 - R) * F / tau_refR) * dt

        # Process vesicle movements based on the computed rates
        if rand_event < transition_RD and R > 0:
            R, D = R - 1, D + 1
        elif rand_event < transition_RD + transition_DR and D > 0:
            D, R = D - 1, R + 1
        elif rand_event < transition_RD + transition_DR + replenish_R and R < R0:
            R, F = R + 1, F - 1 

        # Updating time and storing simulation results
        idx_dt += 1
        t = idx_dt * dt
        times[iteration] = t
        reserve_values[iteration] = R
        docked_values[iteration] = D
        fused_values[iteration] = F
        Ca_pre_values[iteration] = Ca_pre
        Ca_jump_values[iteration] = Ca_jump
        Sigmoid_proba[iteration] = sigmoid(Ca_pre, s)

    # Trimming arrays to actual size
    times = times[:iteration]
    reserve_values = reserve_values[:iteration]
    docked_values = docked_values[:iteration]
    fused_values = fused_values[:iteration]
    Ca_pre_values = Ca_pre_values[:iteration]
    Ca_jump_values = Ca_jump_values[:iteration]
    Sigmoid_proba = Sigmoid_proba[:iteration]

    return times, reserve_values, docked_values, fused_values, Ca_pre_values, Ca_jump_values, Sigmoid_proba, release_times, spike_times

# Function to run a single instance of the simulation
def run_simulation(args):
    D0, R0, F0, tau_adap, delta, tau_D, tau_R, tau_refR, s, decay_rate, jump_size, T_end, dt, max_attempts, quiet_duration, pre_times = args
    return vesicle_release_with_decay(D0, R0, F0, tau_adap, delta, tau_D, tau_R, tau_refR, s, decay_rate, jump_size, T_end, dt, max_attempts, quiet_duration, pre_times)

# Objective function for optimization with averaging for all datasets
def objective_all_datasets(params, all_observed_data, all_observed_time, jump_size, all_frequencies, max_attempts, quiet_duration, n_iter=30):
    tau_D, tau_R, decay_rate, tau_adap, delta, s = params
    total_mse = 0  # Variable to store the total MSE for all datasets

    def process_dataset(args):
        observed_data, observed_time, release_rate = args
        T_end = max_attempts / release_rate  # Total time of simulation
        pre_times = np.linspace(0, max_attempts * (1 / release_rate), max_attempts, endpoint=False)
        mse_values = []
        N = np.round(1 / (release_rate * target_dt))
        dt = 1 / (N * release_rate)  # adapt the dt to the frequency of the spike train

        with ThreadPoolExecutor() as executor:
            futures = [executor.submit(run_simulation, (D0, R0, F0, tau_adap, delta, tau_D, tau_R, tau_refR, s, decay_rate, jump_size, T_end, dt, max_attempts, quiet_duration, pre_times)) for _ in range(n_iter)]
            results = [future.result() for future in futures]

        for times, reserve_values, docked_values, fused_values, _, _, _, _, _ in results:
            simulated_data = np.array(fused_values) / (D0 + R0)
            simulated_data_aligned = []
            for t in observed_time:
                closest_index = np.argmin(np.abs(times - t))
                simulated_data_point = simulated_data[closest_index]
                simulated_data_aligned.append(simulated_data_point)

            mse = np.mean((observed_data - np.array(simulated_data_aligned)) ** 2)
            mse_values.append(mse)

        avg_mse = np.mean(mse_values)
        return avg_mse

    with ThreadPoolExecutor(max_workers=5) as executor:
        mse_results = list(executor.map(process_dataset, zip(all_observed_data, all_observed_time, all_frequencies)))

    total_mse = sum(mse_results)
    mse_values_callback.append(total_mse)
    return total_mse


# Parameters to optimize
D0 = 20                               # Initial docked vesicles
R0 = 30                               # Initial reserve vesicles
tau_D = 20                            # Time constant for vesicle transition from reserve to docked
tau_R =  29                           # Time constant for vesicle transition from docked to reserve
tau_refR = 290                        # Time constant for vesicle replenishment to reserve pool
decay_rate = .15                     # Rate of calcium decay
tau_adap = 1                         # Time constant for calcium adaptation
delta = 4e-02                         # Strength of calcium jump due to AP
s = 0.40                              # Steepness of the release sigmoidal relation

# Parameters fixed
F0 = 0                                # Initial fused vesicles
jump_size = 1                         # Magnitude of calcium jumps
target_dt = 0.01                      # Time step
release_rate = 30.                    # Probability of release per time step (used for Poisson approximation)
max_attempts = 300                    # Max release attempts before quiet period
quiet_duration = 50.                  # Duration of quiet period

# Initial parameters and bounds
initial_params = [tau_D, tau_R, decay_rate, tau_adap, delta, s]
bounds = Bounds([0, 0, 0, 0, 0, 0], [np.inf, np.inf, np.inf, np.inf, np.inf, np.inf])

# Load your CSV files
folder_path = "dataset-Fernandez-Alfonso-2008-25C/preprocessed"
csv_files = natsorted([os.path.join(folder_path, file) for file in os.listdir(folder_path) if file.endswith('.csv')])
labels = ["2 Hz", "5 Hz", "10 Hz", "20 Hz", "30 Hz"]
frequencies = [int(label.split(' ')[0]) for label in labels]

# Global variable to store MSE values
mse_values_callback = []


import numpy as np
import matplotlib.pyplot as plt

def callback(params, all_observed_data, all_observed_time, all_frequencies):
    print(f"Callback for iteration {len(mse_values_callback)}")
    tau_D, tau_R, decay_rate, tau_adap, delta, s = params

    avg_freq = sum(all_frequencies) / len(all_frequencies)
    release_rate = avg_freq
    n_rows = 6
    num_simulations = 20

    print(params)

    for idx, (observed_data, observed_time, release_rate) in enumerate(zip(all_observed_data, all_observed_time, all_frequencies)):
        print(f"Processing dataset {idx+1}/{len(all_observed_data)} with frequency {release_rate} Hz")
        max_attempts = 300
        T_end = max_attempts / release_rate  # Total time of simulation
        pre_times = np.linspace(0, max_attempts * (1 / release_rate), max_attempts, endpoint=False)

        N = np.round(1 / (release_rate * target_dt))
        dt = 1 / (N * release_rate)  # adapt the dt to the frequency of the spike train

        all_simulated_data = []
        
        for sim in range(num_simulations):
            times, reserve_values, docked_values, fused_values, Ca_pre_values, Ca_jump_values, Sigmoid_proba, release_times, spike_times = vesicle_release_with_decay(
                D0, R0, F0, tau_adap, delta, tau_D, tau_R, tau_refR, s, decay_rate, jump_size, T_end, dt, max_attempts, quiet_duration, pre_times)

            simulated_data = np.array(fused_values) / (D0 + R0)
            simulated_data_aligned = []
            for t in observed_time:
                closest_index = np.argmin(np.abs(times - t))
                simulated_data_point = simulated_data[closest_index]
                simulated_data_aligned.append(simulated_data_point)

            all_simulated_data.append(simulated_data_aligned)

        all_simulated_data = np.array(all_simulated_data)
        mean_simulated_data = np.mean(all_simulated_data, axis=0)
        quantile_2nd = np.percentile(all_simulated_data, 25, axis=0)
        quantile_4th = np.percentile(all_simulated_data, 75, axis=0)

        for row in range(n_rows):
            ax = axes[row][idx]
            plt.sca(ax)  # Set the current axis to ax
            ax.clear()
            if row == 0:
                # plot original and simulated data
                observed_time_numeric = np.array(observed_time[1::2], dtype=float)
                observed_data_numeric = np.array(observed_data[1::2], dtype=float)
                mean_simulated_data_numeric = np.array(mean_simulated_data[1::2], dtype=float)
                quantile_2nd_numeric = np.array(quantile_2nd[1::2], dtype=float)
                quantile_4th_numeric = np.array(quantile_4th[1::2], dtype=float)

                plt.plot(observed_time_numeric, observed_data_numeric, 'b', label=f"Observed Data ({release_rate} Hz)")
                plt.plot(observed_time_numeric, mean_simulated_data_numeric, 'r', label=f"Mean Simulated Data ({release_rate} Hz)")
                plt.fill_between(observed_time_numeric, quantile_2nd_numeric, quantile_4th_numeric, color='gray', alpha=0.5, label="2nd to 3rd Quantile Range")
                plt.legend(fontsize='small', markerscale=0.5)
            elif row == 1:
                # plot reserve, docked and release times
                plt.plot(times[1::2], reserve_values[1::2], 'g', label="Reserve Values")
                plt.plot(times[1::2], docked_values[1::2], 'y', label="Docked Values")
                plt.plot(times[1::2], fused_values[1::2], 'y', label="Fused Values")
                plt.scatter(release_times, [0] * len(release_times), c='r', label="Release Times", s=5)
                plt.scatter(spike_times, [2] * len(spike_times), c='C2', label="Spike train", s=5)
                spike_times_exact = np.arange(0, T_end, 1 / release_rate)
                plt.scatter(spike_times_exact, [2 for _ in spike_times], marker="+", color="k", alpha=0.5, label="Spike train, exact", s=10)
            elif row == 2:
                # plot Ca_pre_values
                plt.plot(times[1::2], Ca_pre_values[1::2], 'c', label="Ca_pre Values")
                plt.legend()
            elif row == 3:
                # plot Ca_jump_values and Sigmoid_proba
                plt.plot(times[1::2], Ca_jump_values[1::2], 'm', label="Ca_jump Values")
                plt.plot(times[1::2], Sigmoid_proba[1::2], 'b', label="Sigmoid Proba")
                plt.legend()
            elif row == 4:
                # plot mse_values_callback
                plt.plot(range(len(mse_values_callback)), mse_values_callback, 'b', label="MSE Values Callback")
                plt.legend()
            elif row < n_rows - 1:
                axes[row][idx].sharex(axes[0][idx])  # share time axis for each frequency
            if row == 5:
                z = np.linspace(0, 10, 200)
                y = sigmoid(z, s)
                plt.plot(z, y, label=f'sigmoid(Ca_pre, s={s}')
                plt.xlabel('Ca_pre')
                plt.ylabel('sigmoid(Ca_pre)')
                plt.grid(True)
    plt.tight_layout()
    plt.draw()
    plt.pause(0.1)


# Load all datasets into lists
all_observed_data = []
all_observed_time = []
for file in csv_files:
    df = pd.read_csv(file, header=None, names=['time', 'deltaF spH'])
    all_observed_data.append(df['deltaF spH'])
    all_observed_time.append(df['time'])

# Get screen size using pyautogui
screen_width, screen_height = pyautogui.size()

# Convert screen size to figure size (in inches, assuming 100 DPI)
fig_width = screen_width / 100
fig_height = screen_height / 100

# Create the figure with adjusted size
plt.figure(figsize=(fig_width, fig_height))
axes = [[plt.subplot(6, len(frequencies), idx + 1 + row * len(frequencies)) for idx in range(len(frequencies))] for row in range(6)]

method = 'Nelder-Mead'  # Change this to 'BFGS', 'Powell', 'COBYLA', 'TNC', 'Nelder-Mead' or 'SLSQP'

# Call the minimize function once, outside of the loop
result = minimize(
    objective_all_datasets,
    initial_params,
    args=(all_observed_data, all_observed_time, jump_size, frequencies, max_attempts, quiet_duration),
    method=method,
    callback=lambda params: callback(params, all_observed_data, all_observed_time, frequencies),
    bounds=bounds,
    options={'disp': True, 'maxiter': 500, 'maxfev': 3000}
)

plt.show()
