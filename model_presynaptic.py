import numpy as np
import matplotlib.pyplot as plt

def sigmoid(z, s=1, h=1.0):
    """Sigmoid function with slope s and half-activation at h."""
    return 1 / (1 + np.exp(-s*(z-h)))

def vesicle_release_with_decay(D0, R0, tau_D, tau_R, tau_refR, s, decay_rate, jump_size, T, dt, release_rate):
    """
    Simulate vesicle release dynamics with replenishment, exponential decay, and sigmoid-based release probability.
    """

    R = R0
    D = D0
    Ca_pre = 0

    times = [0]
    reserve_values = [R]
    docked_values = [D]
    Ca_pre_values = [Ca_pre]
    release_times = []

    t = 0

    while t < T:
        Ca_pre *= np.exp(-decay_rate * dt)

        if (int(t * release_rate) != int((t + dt) * release_rate)):
            Ca_pre += jump_size

        rand_event = np.random.uniform(0, 1)

        transition_RD = (D0 - D) * R / tau_D * dt
        transition_DR = (R0 - R) * D / tau_R * dt
        replenish_R = (R0 - R) / tau_refR * dt

        total_rate = transition_RD + transition_DR + replenish_R

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

        if np.random.rand() < sigmoid(Ca_pre, s) * release_rate * dt:
            if D > 0:
                D -= 1
                release_times.append(t)

        t += dt
        times.append(t)
        reserve_values.append(R)
        docked_values.append(D)
        Ca_pre_values.append(Ca_pre)

    return times, reserve_values, docked_values, Ca_pre_values, release_times

# Parameters
D0 = 25
R0 = 30
tau_D = 5.0  # in seconds
tau_R = 45.0  # in seconds
tau_refR = 40.0  # in seconds
s = 2.0
decay_rate = 1.0
jump_size = 20
T = 10
dt = 0.001
release_rate = 1.0

times, reserve_values, docked_values, Ca_pre_values, release_times = vesicle_release_with_decay(D0, R0, tau_D, tau_R, tau_refR, s, decay_rate, jump_size, T, dt, release_rate)

# Plotting
plt.figure(figsize=(12, 8))

plt.subplot(3, 1, 1)
plt.step(times, reserve_values, where='post', label="Reserve Pool (R)")
plt.step(times, docked_values, where='post', label="Docked Pool (D)")
plt.scatter(release_times, [D0] * len(release_times), color='red', label="Release Event", s=15)
plt.xlabel('Time')
plt.ylabel('Vesicle Count')
plt.title('Vesicle Release Dynamics with Replenishment and Sigmoid-Based Release Probability')
plt.grid(True)
plt.legend()

plt.subplot(3, 1, 2)
plt.plot(times, Ca_pre_values, label="Ca_pre (with jumps and decay)")
plt.xlabel('Time')
plt.ylabel('Ca_pre Value')
plt.title('Ca_pre Function with Jumps and Decay')
plt.grid(True)
plt.legend()

plt.tight_layout()
plt.show()
