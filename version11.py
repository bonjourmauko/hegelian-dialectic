import h5py
import numpy as np
import pandas as pd


# Dialectical functions


def psi_1(x):
    """Pure being and pure nothing"""
    return x - 1j * x


def psi_2(x):
    """Coming-to-be and ceasing-to-be"""
    if x == 0:
        return np.exp(1) + np.exp(1j)

    return psi_2(0) * (np.exp(x) + np.exp(-1j * x))


def psi_3(x):
    """Identity of being and nothing"""
    if x == 0:
        return np.exp(1) + np.exp(1j)

    return (1 / psi_2(0)) * (1 / np.exp(x) + 1 / np.exp(-1j * x))


def psi_4(x):
    """Vanishing of being and nothing"""
    return 10**15 + 1j / 10**15


def psi_5(x):
    """Vanishing into opposite"""
    return 1 / 10**15 + 1 / (1j / 10**15)


def psi_6(x):
    """Mutual vanishing"""
    return -psi_5(x)


def psi_7(x):
    """Becoming - Being and nothing"""
    return -x - 1j * x


def psi_8(x):
    """Becoming - Difference"""
    return x + -1j * x


def psi_9(x):
    """Becoming - Identity"""
    return -x + 1j * x


def psi_10(x):
    """Becoming as ceaseless unrest"""
    return np.cos(x) - 1j * np.sin(x)


def psi_11(x):
    """Becoming as quiescent result"""
    return np.cos(x) ** 2 + 1j * np.sin(x) ** 2


# Group them for Markov choising

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
    psi_10,
    psi_11,
]

# Define dependencies

dependencies = {
    1: [],
    2: [],
    3: [],
    4: [],
    5: [],
    6: [],
    7: [],
    8: [],
    9: [],
    10: [],
    11: [],
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

x_vals = np.linspace(0, 200 * np.pi, 1000000, dtype=np.complex128)

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

df.to_csv("data.v11.csv")

df.to_hdf("data.h5", "v11")
