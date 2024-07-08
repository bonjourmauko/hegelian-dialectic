"""Example of Hegel's logic as a wave function"""

import h5py
import numpy as np
import pandas as pd

# Total number of iterations
n = 10**5

# Slice
s = 10**4

# Number of oscillations
p = n / s * np.pi


def psi_1(x):
    """Being"""
    return np.cos(x)


def psi_2(x):
    """Nothing"""
    return 1j * np.sin(x)


def psi_3(x):
    """Becoming — Difference — Identity — ?"""
    return psi_1(x) + psi_2(x)


def psi_4(x):
    """Passing"""
    return psi_2(psi_1(x))


def psi_5(x):
    """Arising"""
    return psi_1(psi_2(x))


def psi_6(x):
    """Equilibrium"""
    return psi_5(psi_3(x)) + psi_4(psi_3(x))


def psi_7(x):
    """Emerging"""
    return psi_3(psi_1(x))


def psi_8(x):
    """Dissolving"""
    return psi_6(psi_2(x))


def psi_9(x):
    """Harmony"""
    return psi_8(psi_3(x)) + psi_7(psi_3(x))


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


# Normalise data
def normalise(vector):
    return (vector - vector.min()) / (vector.max() - vector.min())


for m in range(1, s + 1):
    index = int(n * m / s)

    print(index)

    # Generate x values
    x_vals = np.linspace(0, p + 1/p, index, dtype=np.complex128)

    # Generate steps

    values = []

    # Save the state differential

    states = []

    spinor_x = []
    spinor_y = []

    state = None

    sign = None

    for i, x in enumerate(x_vals):
        if i == 0:
            state = 0
            states.append(state)
            spinor_x.append(np.random.choice(np.linspace(-1/2, 1/2, 2)))
            spinor_y.append(np.random.choice(np.linspace(-1j/2, 1j/2, 2)))

        else:
            state = next_state(state)
            states.append(state)
            delta = state - states[i - 1]

            if delta == 0:
                sign = 1

            else:
                sign = np.sign(delta)

            spinor_x.append(
                sign
                * np.sign(spinor_x[i - 1])
                * np.random.choice(np.linspace(-1/2, 1/2, 2))
            )
            spinor_y.append(
                sign
                * np.sign(spinor_y[i - 1])
                * np.random.choice(np.linspace(-1j/2, 1j/2, 2))
            )

        values.append(psi(state, x) * (spinor_x[i] + spinor_y[i]))

    # Evaluate psi(x) for these x values

    psi_vals = np.array(values)

    data_x = normalise(psi_vals.real)
    data_y = normalise(psi_vals.imag)

    # Extract real and imaginary parts for mapping x -> x, y -> y.real, z -> y.imag

    df = pd.DataFrame({"Ψ(x)": data_x, "Ψ*(x)": data_y})
    lower = df.quantile(0.0001)
    upper = df.quantile(0.9999)
    df_clipped = df.clip(lower, upper, axis=1)

    # Plotting

    # Save to disk

    df_clipped.to_hdf("data.h5", key=f"v12n{m}")
