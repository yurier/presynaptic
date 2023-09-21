import numpy as np
import matplotlib.pyplot as plt

# Define the sigmoid function
def sigmoid(z, s=1, h=4):
    """
    Sigmoid function with slope s and half-activation at h.
    
    Parameters:
    - z: Input to the function
    - s: Slope of the sigmoid
    - h: Half-activation point
    
    Returns:
    - Value of the sigmoid function at z
    """
    return 1 / (1 + np.exp(-s*(z-h)))

def vesicle_release_with_decay(D0, R0, tau_D, tau_R, tau_refR, s, decay_rate, jump_size, T, dt, release_rate, max_attempts, quiet_duration):
    """
    Simulate vesicle release dynamics considering vesicle replenishment, calcium-driven exponential decay, 
    and a sigmoid-based vesicle release probability.
    
    Parameters:
    - D0: Initial count of docked vesicles
    - R0: Initial count of reserve vesicles
    - tau_D: Transition rate from reserve to docked
    - tau_R: Transition rate from docked to reserve
    - tau_refR: Replenishment rate for the reserve pool
    - s: Slope of the sigmoid
    - decay_rate: Rate at which calcium decays
    - jump_size: Increase in calcium upon a release event
    - T: Total simulation time
    - dt: Time step of the simulation
    - release_rate: Rate at which vesicles are released
    
    Returns:
    - times: A list of time points of the simulation
    - reserve_values: Corresponding vesicle count in the reserve pool
    - docked_values: Corresponding vesicle count in the docked pool
    - Ca_pre_values: Calcium concentration over time
    - release_times: Time points at which vesicle releases occurred
    """
    
    # Initialize vesicle counts and calcium concentration
    R = R0
    D = D0
    Ca_pre = 0

    # Lists to store simulation results
    times = [0]
    reserve_values = [R]
    docked_values = [D]
    Ca_pre_values = [Ca_pre]
    release_times = []

    t = 0

    # Initialize a counter for release attempts
    release_attempts = 0

    # Initialize a flag for the quiet period
    in_quiet_period = False
    
    # Initialize timer for the quiet period
    quiet_timer = 0

    # Main simulation loop
    while t < T+ quiet_duration:
        # Compute the time for the next potential vesicle release
        next_release_time = (np.floor(t * release_rate) + 1) / release_rate 
        
        # Process each time step until the next release time or end of simulation
        while t < next_release_time and t < T + quiet_duration:  # Adjusted condition here as well
            if in_quiet_period:
                quiet_timer += dt
                if quiet_timer >= quiet_duration:
                    # Exit the quiet period
                    in_quiet_period = False
                    quiet_timer = 0
                    release_attempts = 0  # Reset the counter

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
            if rand_event < transition_RD:
                if R > 0:
                    R -= 1
                    D += 1
            elif rand_event < transition_RD + transition_DR:
                if D > 0:
                    D -= 1
                    R += 1
            elif rand_event < transition_RD + transition_DR + replenish_R:
                if R < R0:
                    R += 1

            # Move forward in time
            t += dt
            times.append(t)
            reserve_values.append(R)
            docked_values.append(D)
            Ca_pre_values.append(Ca_pre)

        # Attempt to release only if not in a quiet period
        if not in_quiet_period:
            release_attempts += 1
            Ca_pre += jump_size
            if np.random.rand() < sigmoid(Ca_pre, s):
                if D > 0:
                    D -= 1
                    release_times.append(t)

            # Check if we've reached the maximum number of attempts
            if release_attempts >= max_attempts:
                in_quiet_period = True


    return times, reserve_values, docked_values, Ca_pre_values, release_times

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

# Execute the simulation
times, reserve_values, docked_values, Ca_pre_values, release_times = vesicle_release_with_decay(
    D0, R0, tau_D, tau_R, tau_refR, s, decay_rate, jump_size, T, dt, release_rate, max_attempts, quiet_duration)

# Plotting the simulation results
plt.figure(figsize=(12, 8))

# Plot vesicle dynamics
plt.subplot(3, 1, 1)
plt.step(times, reserve_values, where='post', label="Reserve Pool (R)")
plt.step(times, docked_values, where='post', label="Docked Pool (D)")
plt.scatter(release_times, [D0] * len(release_times), color='red', label="Release Event", s=15)
plt.xlabel('Time')
plt.ylabel('Vesicle Count')
plt.title('Vesicle Release Dynamics with Replenishment and Sigmoid-Based Release Probability')
plt.grid(True)
plt.legend()

# Plot calcium dynamics
plt.subplot(3, 1, 2)
plt.plot(times, Ca_pre_values, label="Ca_pre (with jumps and decay)")
plt.xlabel('Time')
plt.ylabel('Ca_pre Value')
plt.title('Ca_pre Function with Jumps and Decay')
plt.grid(True)
plt.legend()

plt.tight_layout()
plt.show()
