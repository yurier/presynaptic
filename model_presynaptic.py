import numpy as np
import matplotlib.pyplot as plt

def sigmoid(z, s, h):
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
quiet_duration = 100                  # Duration of quiet period

#last optimization [3.22059588e+01 1.82230449e+01 1.31264672e+01 3.46225756e-01 7.89550730e+00 6.48332889e+00 5.25686085e-02 2.63467268e-02]
# initial_params = [tau_D, tau_R, tau_refR, s, h, decay_rate, tau_adap, delta]

# Time array to represent when pre-synaptic spikes occur
T_end = (max_attempts/release_rate)  # Total time of simulation
pre_times = np.linspace(0, max_attempts * (1/release_rate), max_attempts, endpoint=False)


# Run Simulation
times, reserve_values, docked_values, Ca_pre_values, Ca_jump_values, Sigmoid_proba, release_times = vesicle_release_with_decay(
    D0, R0, tau_adap, delta, tau_D, tau_R, tau_refR, h, s, decay_rate, jump_size, T_end, dt,  max_attempts, quiet_duration, pre_times)

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
plt.plot(times, reserve_values, label="Reserve Vesicles")
plt.plot(times, docked_values, label="Docked Vesicles")
plt.scatter(release_times, [R0 for _ in release_times],color='red', label="Release Events")

plt.ylabel("Vesicle Counts")
plt.legend()

plt.subplot(4,1,4)
plt.scatter(release_times, [1 for _ in release_times],color='red', label="Release Events")
plt.xlim([0, T_end])
plt.xlabel("Time")
plt.ylabel("Release Events")
plt.legend()

plt.tight_layout()
plt.show()

# Plotting vesicle dynamics for different release rates
plt.figure(figsize=(12, 8))
plt.subplot(1, 1, 1)
for release_rate in [2, 5, 10, 20, 30]:
    # Time array to represent when pre-synaptic spikes occur
    T_end = (max_attempts/release_rate)  # Total time of simulation
    pre_times = np.linspace(0, max_attempts * (1/release_rate), max_attempts, endpoint=False)
    times, reserve_values, docked_values, Ca_pre_values, Ca_jump_values, Sigmoid_proba, release_times = vesicle_release_with_decay(
    D0, R0, tau_adap, delta, tau_D, tau_R, tau_refR, h, s, decay_rate, jump_size, T_end, dt,  max_attempts, quiet_duration, pre_times)
    plt.step(times, 1-docked_values/np.max(docked_values), where='post', label=f"Docked Pool (D) with release rate {release_rate}")
    plt.xlabel('Time')
    plt.ylabel('Vesicle Count')
    plt.title('Vesicle Release Dynamics with Replenishment for Different Release Rates')
    plt.grid(True)
    plt.legend()

plt.show()
