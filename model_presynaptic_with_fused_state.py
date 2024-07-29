import numpy as np
import matplotlib.pyplot as plt
from natsort import natsorted
import os
import pandas as pd

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



# Parameters to optimize
D0 = 20                               # Initial docked vesicles
R0 = 30                               # Initial reserve vesicles
tau_D = 20                            # Time constant for vesicle transition from reserve to docked
tau_R =  29                           # Time constant for vesicle transition from docked to reserve
tau_refR = 290*.25                    # Time constant for vesicle replenishment to reserve pool
decay_rate = .15*.25                  # Rate of calcium decay
tau_adap = 1                          # Time constant for calcium adaptation
delta = 4e-02                         # Strength of calcium jump due to AP
s = 0.38                              # Steepness of the release sigmoidal relation

# Parameters fixed
F0 = 0                                # Initial fused vesicles
jump_size = 1*.25                         # Magnitude of calcium jumps
dt = 0.01                             # Time step
release_rate = 30.                    # Probability of release per time step (used for Poisson approximation)
max_attempts = 300                    # Max release attempts before quiet period
quiet_duration = 50.                  # Duration of quiet period


# Time array to represent when pre-synaptic spikes occur
T_end = (max_attempts/release_rate)  # Total time of simulation
pre_times = np.linspace(0, max_attempts * (1/release_rate), max_attempts, endpoint=False)


""" # Run Simulation
times, reserve_values, docked_values, fused_values, Ca_pre_values, Ca_jump_values, Sigmoid_proba, release_times, spike_times = vesicle_release_with_decay(
    D0, R0, F0, tau_adap, delta, tau_D, tau_R, tau_refR, s, decay_rate, jump_size, T_end, dt,  max_attempts, quiet_duration, pre_times)

# Plot Results
plt.figure(figsize=(14,10))

plt.subplot(4,1,1)
plt.plot(times, Ca_pre_values, label="Ca_pre")
plt.plot(times, Ca_jump_values, label="Ca_jump")
plt.ylabel("Calcium Levels")
plt.legend()

plt.subplot(4,1,2)
plt.plot(times, Sigmoid_proba, label="Release Probability")
plt.ylabel("Probability")
plt.legend()

plt.subplot(4,1,3)
plt.plot(times, fused_values, label="Fused Vesicles")
plt.plot(times, reserve_values, label="Reserve Vesicles")
plt.plot(times, docked_values, label="Docked Vesicles")
plt.scatter(release_times, [R0 for _ in release_times],color='red', label="Release Events")

plt.ylabel("Vesicle Counts")
plt.legend()

plt.subplot(4,1,4)
plt.scatter(release_times, [1 for _ in release_times],color='red', label="Release Events")
plt.scatter(spike_times, [0.98 for _ in spike_times], color='C2', label="Spike train")
spike_times_exact = np.arange(0, T_end, 1/release_rate)
plt.scatter(spike_times_exact, [0.98 for _ in spike_times], marker="+", color="k", alpha=0.5, label="Spike train, exact")
plt.xlim([0, T_end])
plt.ylim([0.95, 1.05])
plt.yticks([])
plt.xlabel("Time")
# plt.ylabel("Release Events")
plt.legend()

plt.tight_layout()
plt.show()

 """
# Load your CSV files
folder_path = "dataset-Fernandez-Alfonso-2008-35C/preprocessed"
csv_files = natsorted([os.path.join(folder_path, file) for file in os.listdir(folder_path) if file.endswith('.csv')])
labels = ["2 Hz", "5 Hz", "10 Hz", "20 Hz", "30 Hz"]
frequencies = [int(label.split(' ')[0]) for label in labels]

# Load all datasets into lists
all_observed_data = []
all_observed_time = []
for file in csv_files:
    df = pd.read_csv(file, header=None, names=['time', 'deltaF spH'])
    all_observed_data.append(df['deltaF spH'])
    all_observed_time.append(df['time'])


# Plotting vesicle dynamics for different release rates
plt.figure(figsize=(12*2*.75, 3*3*.75))
plt.subplot(3, 5, 1)
count = 1
count2 = 1
for release_rate in [2, 5, 10, 20, 30]:
    # Time array to represent when pre-synaptic spikes occur
    T_end = (max_attempts/release_rate)  # Total time of simulation
    pre_times = np.linspace(0, max_attempts * (1/release_rate), max_attempts, endpoint=False)
    times, reserve_values, docked_values, fused_values, Ca_pre_values, Ca_jump_values, Sigmoid_proba, release_times, spike_times = vesicle_release_with_decay(D0, R0, F0, tau_adap, delta, tau_D, tau_R, tau_refR, s, decay_rate, jump_size, T_end, dt,  max_attempts, quiet_duration, pre_times)
    plt.subplot(3, 5, count)
    plt.plot(all_observed_time[count-1],all_observed_data[count-1])
    #count2=1+count2
    #count=1+count

    plt.step(times, fused_values/(D0+R0), where='post', label="Fused")
    if release_rate == 2:
        plt.ylabel('Vesicle Count')

    plt.grid(True)
    plt.title(f"{release_rate} Hz, 35C")
    plt.legend()
    # plot Ca_pre_values
    plt.subplot(3, 5, count+5)
    #count=1+count
    plt.plot(times[1::2], Ca_pre_values[1::2], 'c')
    if release_rate == 2:
        plt.ylabel('Ca_pre')

    # plot Ca_jump_values and Sigmoid_proba
    plt.subplot(3, 5, count+10)
    count=1+count

    plt.plot(times[1::2], Ca_jump_values[1::2], 'm', label="Ca_jump Values")
    plt.plot(times[1::2], Sigmoid_proba[1::2], 'b', label="Sigmoid Proba")
    plt.xlabel('Time')
    plt.legend()
plt.show()
