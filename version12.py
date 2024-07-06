"""Example of Hegel's logic as a wave function"""

import h5py
import numpy as np
import pandas as pd

# Total number of iterations
n = 10**7

# Number of oscillations
p = (n / 10**3) * np.pi


def psi_1(x):
    """Being"""
    x = x.real
    k = x % np.pi
    real_part = np.cos(k)
    imag_part = np.sin(k)
    return real_part + 1j * imag_part


def psi_2(x):
    """Nothing"""
    return psi_1(x) + psi_3(x)


def psi_3(x):
    """Becoming — Difference — Identity"""
    x = x.real
    k = x % np.pi
    real_part = np.cos(x)
    imag_part = -np.sin(x)
    return real_part + 1j * imag_part


def psi_4(x):
    """Passing"""
    return psi_2(psi_1(x))


def psi_5(x):
    """Arising"""
    return psi_1(psi_2(x))


def psi_6(x):
    """Equilibrium"""
    return psi_4(psi_3(x)) - psi_5(psi_3(x))


def psi_7(x):
    """Emerging"""
    return psi_3(psi_1(x))


def psi_8(x):
    """Dissolving"""
    return psi_6(psi_2(x))


def psi_9(x):
    """Harmony"""
    return psi_8(psi_3(x)) - psi_7(psi_3(x))


# Group them for Markov chaining

psi_functions = [
    psi_1,
    psi_2,
    psi_3,
    psi_4,
    psi_5,
    psi_6,
    psi_7,
    psi_8,
    psi_9,
]

# Define rules

rules = {
    (1,): [1, 2],
    (2,): [1, 2, 3],
    (3,): [1, 2, 3, 4],
    (4,): [3, 4, 5],
    (5,): [3, 4, 5, 6],
    (6,): [4, 5, 6, 7],
    (7,): [6, 7, 8],
    (8,): [6, 7, 8, 9],
    (9,): [7, 8, 9],
}

# Initialize transition matrix

n_states = len(psi_functions)

transition_matrix = np.zeros((n_states, n_states))

# Populate transition matrix with appropriate probabilities

for from_states, to_states in rules.items():
    prob = 1 / len(to_states)  # Equiprobable transitions
    for from_state in from_states:
        for to_state in to_states:
            transition_matrix[from_state - 1, to_state - 1] = prob

# Ensure rows sum up to 1

row_sums = transition_matrix.sum(axis=1)

transition_matrix /= row_sums[:, np.newaxis]

# Function to simulate one step of the Markov chain


def next_state(current_state):
    return np.random.choice(range(n_states), p=transition_matrix[current_state])


# Define the composite function Ψ(x) with stochastic selection


def psi(state, x):
    return psi_functions[state](x)


# Generate x values
x_vals = np.linspace(-p, p, n, dtype=np.complex128)

# Generate steps

values = []

# Save the state differential

states = []

state = None

for i, x in enumerate(x_vals):
    if state is None:
        state = 0
        states.append(state)

    else:
        state = next_state(state)
        states.append(state - states[i - 1])

    values.append(psi(state, x))

# Evaluate psi(x) for these x values

psi_vals = np.array(values)


# Normalise data
def normalise(vector):
    return (vector - vector.min()) / (vector.max() - vector.min())


data_x = normalise(psi_vals.real)
data_y = normalise(psi_vals.imag)
data_z = np.array(states, dtype=np.int8)

# Extract real and imaginary parts for mapping x -> x, y -> y.real, z -> y.imag

df = pd.DataFrame({"Ψ(x)": data_x, "Ψ*(x)": data_y, "Δs": data_z, "x": x_vals.real})

# Plotting

# Save to disk

df.to_csv(f"data.v12.n{n}.csv")

df.to_hdf("data.h5", f"v10n{n}")
