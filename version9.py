import h5py
import numpy as np
import pandas as pd


def spin():
    return np.random.choice([-1, 0, 1])


# Dialectical functions


def psi_1(x):
    return np.cos(x) + 1j * np.sin(x) * spin()


def psi_2(x):
    return np.cos(x) - 1j * np.sin(x) * spin()


def psi_3(x):
    return psi_1(x) + psi_2(x)


def psi_4(x):
    return psi_2(psi_1(x))


def psi_5(x):
    return psi_1(psi_2(x))


def psi_6(x):
    return psi_4(psi_3(x)) * psi_5(psi_3(x))


def psi_7(x):
    return psi_3(psi_1(x))


def psi_8(x):
    return psi_6(psi_2(x))


def psi_9(x):
    return psi_7(psi_3(x)) * psi_8(psi_3(x))


# Group them for Markon choising

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

# Define dependencies

dependencies = {
    1: list(range(3, 10)),
    2: list(range(3, 10)),
    3: list(range(4, 10)),
    4: list(range(6, 10)),
    5: list(range(6, 10)),
    6: list(range(7, 10)),
    7: list(range(9, 10)),
    8: list(range(9, 10)),
    9: list(range(10, 10)),
}

# Initialize transition matrix

n_states = len(dependencies)

transition_matrix = np.zeros((n_states, n_states))

# Populate transition matrix with appropriate probabilities

for i in range(1, n_states + 1):
    valid_transitions = [j for j in range(1, n_states + 1) if j not in dependencies[i]]
    num_valid_transitions = len(valid_transitions)

    if num_valid_transitions > 0:
        prob = 1 / num_valid_transitions

        for j in valid_transitions:
            transition_matrix[i - 1, j - 1] = prob

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

x_vals = np.linspace(-1000 * np.pi, 1000 * np.pi, 1000000, dtype=np.complex128)

# Generate steps

values = []

state = None

for i, x in enumerate(x_vals):
    if state is None:
        state = 0

    else:
        state = next_state(state)

    values.append(psi(state, x))

# Evaluate psi(x) for these x values

psi_vals = np.array(values)

# Normalise data

data = np.vstack([x_vals.real, psi_vals.real, psi_vals.imag]).T

normalised = (data - data.min()) / (data.max() - data.min())

# Extract real and imaginary parts for mapping x -> x, y -> y.real, z -> y.imag

df = pd.DataFrame(normalised, columns=("x", "Ψ(x)", "Ψ*(x)"))

# Plotting

# Save to disk

df.to_csv("data.v9.csv")

df.to_hdf("data.h5", "v9")
