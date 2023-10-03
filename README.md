# Vesicle Release Dynamics with Replenishment and Sigmoid-Based Release Probability

This model simulates the dynamics of vesicle release in a synapse based on two primary vesicle pools - a reserve pool `R` and a docked pool `D` - and the influence of $Ca^{2+}$ concentration, denoted as $Ca_{pre}$.

## Model Overview

![Figure](/output.png)

## Simulation Details

The simulation uses a combination of Gillespie's algorithm and Euler's method for numerical integration to simulate vesicle dynamics, calcium concentration decay, vesicle release events, and the adaptation of calcium jumps.

### 1. Model Variables

- `R(t)`: Number of vesicles in the reserve pool at time `t`.
- `D(t)`: Number of vesicles in the docked pool at time `t`.
- `Ca_pre(t)`: `Ca^2+` concentration at the presynaptic terminal at time `t`.

### 2. Initial Conditions

- `D_0`: Initial number of vesicles at `D` (given as 25).
- `R_0`: Initial number of vesicles at `R` (given as 30).

### 3. Rate Constants

- `τ_D`: Time constant for transition from `R` to `D` (given as 5 seconds).
- `τ_R`: Time constant for transition from `D` to `R` (given as 45 seconds).
- `τ_refR`: Time constant for replenishment to `R` (given as 40 seconds).

### 4. Vesicle Transitions

- **From reserve (R) to docked (D)**:
  $$\frac{dR}{dt} = -\frac{(D_0 - D(t)) \cdot R(t)}{\tau_D}$$
  $$\frac{dD}{dt} = \frac{(D_0 - D(t)) \cdot R(t)}{\tau_D}$$

- **From docked (D) to reserve (R)**:
  $$\frac{dD}{dt} = -\frac{(R_0 - R(t)) \cdot D(t)}{\tau_R}$$
  $$\frac{dR}{dt} = \frac{(R_0 - R(t)) \cdot D(t)}{\tau_R}$$

- **Replenishment to reserve (R)**:
  $$\frac{dR}{dt} = \frac{R_0 - R(t)}{\tau_{refR}}$$

### 5. $Ca^{2+}$ Dynamics

The concentration $Ca_{pre}$ decays exponentially and has synchronous jumps based on a certain release rate:

$$Ca_{pre}(t+dt) = Ca_{pre}(t) \cdot e^{-\mbox{decay rate} \cdot dt}$$

With jumps at regular intervals determined by the release rate.

### 6. Vesicle Release Probability

Vesicle release from D depends on a sigmoid function of $Ca_{pre}$:

$$P_{release}(t) = \frac{1}{1 + e^{-s \times (Ca_{pre}(t) - 1)}}$$

Where \(s\) (given as 2) determines the slope of the sigmoid.


## How to Use

1. Clone this repository.
2. Place your figure in the root directory and update the path in this README.
3. Run the provided Python script to generate simulation results.

## TODO
- [ ] Make it neuron? (maybe too complicated, but useful if we want to scale it easily)
- [ ] Fit to data (The Kinetics of Synaptic Vesicle Pool Depletion at CNS Synaptic Terminals)
![Figure](/Fernandez-Alfonso-2008_dataset.png)
 
## Some papers

- [Joselevitch, Christina, and David Zenisek. "Direct observation of vesicle transport on the synaptic ribbon provides evidence that vesicles are mobilized and prepared rapidly for release." Journal of Neuroscience 40.39 (2020): 7390-7404.](https://www.jneurosci.org/content/40/39/7390)
- [Hallermann, Stefan. "Calcium channels for endocytosis." The Journal of Physiology 592.Pt 16 (2014): 3343.](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC4229332/)
- [Fernández-Alfonso, Tomás, and Timothy A. Ryan. "The kinetics of synaptic vesicle pool depletion at CNS synaptic terminals." Neuron 41.6 (2004): 943-953.](https://doi.org/10.1016/S0896-6273(04)00113-8)


## Parameter Optimization

The `parameter_optimization_simultaneous.py` script offers a way to fine-tune the vesicle release model parameters based on observed data. By leveraging the Nelder-Mead optimization method, the script aims to minimize the mean squared error (MSE) between the simulated and observed data. For a more interactive experience, the optimization process is visualized in real-time using PyQt5 and pyqtgraph.

### How to Use

1. Ensure you have the necessary libraries installed: 
   ```bash
   pip install numpy pandas scipy pyqtgraph PyQt5
