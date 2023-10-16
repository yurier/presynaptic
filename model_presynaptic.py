# Import necessary libraries
import numpy as np
import matplotlib.pyplot as plt

# Defining the sigmoid function used for vesicle release probability
def sigmoid(z, s, h):
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

# Defining the main simulation function for vesicle release with decay
def vesicle_release_with_decay(D0, R0, tau_adap, delta, tau_D, tau_R, tau_refR, h, s, decay_rate, jump_size, T, dt, release_rate, max_attempts, quiet_duration):
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
    
    # Initializing vesicle counts, calcium concentration, and Ca_jump
    R, D, Ca_pre, Ca_jump = R0, D0, 0, 1  # Initialize Ca_jump to 0 or any desired value

    # Initializing lists to store simulation results
    times, reserve_values, docked_values, Ca_pre_values, Ca_jump_values, Sigmoid_proba, release_times = [0], [R], [D], [Ca_pre], [Ca_jump], [0], []

    t, release_attempts, in_quiet_period, quiet_timer, idx_dt = 0, 0, False, 0, 0  # Additional initializations

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

            # Update Ca_jump using Euler's method for numerical integration
            dCa_jump = (1 - Ca_jump) * tau_adap - (delta * Ca_jump * Ca_pre)
            Ca_jump += dCa_jump * dt  # Update Ca_jump

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
            # t += dt
            idx_dt += 1
            t = idx_dt*dt
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
                print(rand, sigmoid(Ca_pre, s, h))
                if D > 0:
                    D -= 1
                    release_times.append(t)

        # Check if we've reached the maximum number of attempts
        if release_attempts == max_attempts:
            in_quiet_period = True


    return times, reserve_values, docked_values, Ca_pre_values, Ca_jump_values, Sigmoid_proba,release_times

# Setting the parameters for the simulation
D0 = 25
R0 = 30
tau_D = 3.12883049e+01#2.68104323e+01#2.60381944e+01#2.51406722e+01  
tau_R = 1.58603292e+01#2.01812934e+01#1.97873162e+01  
tau_refR = 1.89092587e+01#2.22210394e+01#2.17659560e+01
s = 5.52438840e-01#7.90609246e-01#0.8
h = 6.62098313e+00#5.26474562e+00#6.08386430e+00
decay_rate = 5.28309299e+00
jump_size = 1
dt = 0.05
release_rate = 30
max_attempts=300
T = max_attempts/release_rate
quiet_duration=200
# Parameter for Ca_jump adaptation
tau_adap = 5.36337317e-02#5.24849781e-02#5.24280854e-02#5.12249414e-02 
delta = 2.20954120e-02#2.09973392e-02#2.09796610e-02#1.98393158e-02

# Execute the simulation
times, reserve_values, docked_values, Ca_pre_values, Ca_jump_values, Sigmoid_proba, release_times = vesicle_release_with_decay(
    D0, R0, tau_adap, delta, tau_D, tau_R, tau_refR, h, s, decay_rate, jump_size, T, dt, release_rate, max_attempts, quiet_duration)

[3.12883049e+01 1.58603292e+01 1.89092587e+01 5.52438840e-01
 6.62098313e+00 5.28309299e+00 5.36337317e-02 2.20954120e-02]
# Plotting the simulation results
# plt.figure(figsize=(12, 8))

# Plot vesicle dynamics
# plt.subplot(3, 1, 1)
# plt.step(times, reserve_values, where='post', label="Reserve Pool (R)")
# plt.step(times, docked_values/np.max(docked_values), where='post', label="Docked Pool (D)")
#plt.scatter(release_times, [D0] * len(release_times), color='red', label="Release Event", s=15)
# plt.xlabel('Time')
# plt.ylabel('Vesicle Count')
# plt.title('Vesicle Release Dynamics with Replenishment')
# plt.grid(True)
# plt.legend()

# # Plot calcium dynamics
# plt.subplot(3, 1, 2)
# plt.plot(times, Ca_pre_values, label="Ca_pre (with jumps and decay)")
# plt.xlabel('Time')
# plt.ylabel('Ca_pre Value')
# plt.title('Ca_pre Function with Jumps and Decay')
# plt.grid(True)
# plt.legend()

# # Plot calcium dynamics
# plt.subplot(3, 1, 3)
# plt.plot(times, Ca_jump_values, label="Ca_jump adaptation")
# plt.xlabel('Time')
# plt.ylabel('Ca_pre Value')
# plt.title('Ca_pre Function with Jumps and Decay')
# plt.grid(True)
# plt.legend()
# plt.tight_layout()

# # Plot calcium dynamics
# plt.subplot(3, 1, 3)
# plt.plot(times, Sigmoid_proba, label="Probability")
# plt.xlabel('Time')
# plt.ylabel('prob(Ca) and adaptation')
# plt.title('Sigmoid probability and Ca2+ jump adaptation')
# plt.grid(True)
# plt.legend()
# plt.tight_layout()


# plt.show()

# # Generate a range of values to plot
# z_values = np.linspace(-10, 10, 400)  # Generate 400 values between -10 and 10

# # Compute the corresponding sigmoid values
# sigmoid_values = sigmoid(z_values, s, h)  # Use s=1 and h=4

# # Plot the sigmoid function
# plt.figure(figsize=(8, 6))
# plt.plot(z_values, sigmoid_values, label=f"Sigmoid with s=1, h=4")
# plt.axvline(0, color='gray', lw=0.5)  # Add a vertical line at x=0
# plt.axhline(0.5, color='gray', lw=0.5)  # Add a horizontal line at y=0.5
# plt.xlabel('z')
# plt.ylabel('sigmoid(z)')
# plt.title('Sigmoid Function')
# plt.legend()
# plt.grid(True)
# plt.show()

# Plotting the simulation results
plt.figure(figsize=(12, 8))

# Plot vesicle dynamics
<<<<<<< HEAD
plt.subplot(3, 1, 1)

for release_rate in [2,5,10,20,30]:
    # Execute the simulation
    times, reserve_values, docked_values, Ca_pre_values, Ca_jump_values, Sigmoid_proba, release_times = vesicle_release_with_decay(
        D0, R0, tau_adap, delta, tau_D, tau_R, tau_refR, h, s, decay_rate, jump_size, T, dt, release_rate, max_attempts, quiet_duration)

    # Plotting the simulation results
    # plt.figure(figsize=(12, 8))

    # Plot vesicle dynamics
    # plt.subplot(3, 1, 1)
    # plt.step(times, reserve_values, where='post', label="Reserve Pool (R)")
    plt.step(times, 1-docked_values/np.max(docked_values), where='post', label="Docked Pool (D)")
    #plt.scatter(release_times, [D0] * len(release_times), color='red', label="Release Event", s=15)
    plt.xlabel('Time')
    plt.ylabel('Vesicle Count')
    plt.title('Vesicle Release Dynamics with Replenishment')
    plt.grid(True)
    plt.legend()

plt.show()
=======
ax_1 = plt.subplot(3, 1, 1)
plt.step(times, reserve_values, where='post', label="Reserve Pool (R)")
plt.step(times, docked_values, where='post', label="Docked Pool (D)")
plt.scatter(release_times, [D0] * len(release_times), color='red', label="Release Event", s=15)
plt.xlabel('Time')
plt.ylabel('Vesicle Count')
plt.title('Vesicle Release Dynamics with Replenishment')
plt.grid(True)
plt.legend()

# Plot calcium dynamics
plt.subplot(3, 1, 2, sharex=ax_1)
plt.plot(times, Ca_pre_values, label="Ca_pre (with jumps and decay)")
plt.xlabel('Time')
plt.ylabel('Ca_pre Value')
plt.title('Ca_pre Function with Jumps and Decay')
plt.grid(True)
plt.legend()

# Plot calcium dynamics
plt.subplot(3, 1, 3, sharex=ax_1)
plt.plot(times, Ca_jump_values, label="Ca_jump adaptation")
plt.xlabel('Time')
plt.ylabel('Ca_pre Value')
plt.title('Ca_pre Function with Jumps and Decay')
plt.grid(True)
plt.legend()
plt.tight_layout()

# Plot calcium dynamics
plt.subplot(3, 1, 3)
plt.plot(times, Sigmoid_proba, label="Probability")
plt.xlabel('Time')
plt.ylabel('prob(Ca) and adaptation')
plt.title('Sigmoid probability and Ca2+ jump adaptation')
plt.grid(True)
plt.legend()
plt.tight_layout()


plt.show()

# Generate a range of values to plot
z_values = np.linspace(-10, 10, 400)  # Generate 400 values between -10 and 10

# Compute the corresponding sigmoid values
sigmoid_values = sigmoid(z_values, s, h)  # Use s=1 and h=4

# Plot the sigmoid function
plt.figure(figsize=(8, 6))
plt.plot(z_values, sigmoid_values, label=f"Sigmoid with s=1, h=4")
plt.axvline(0, color='gray', lw=0.5)  # Add a vertical line at x=0
plt.axhline(0.5, color='gray', lw=0.5)  # Add a horizontal line at y=0.5
plt.xlabel('z')
plt.ylabel('sigmoid(z)')
plt.title('Sigmoid Function')
plt.legend()
plt.grid(True)
plt.show()
>>>>>>> main
