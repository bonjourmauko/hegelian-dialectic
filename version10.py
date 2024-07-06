import h5py
import numpy as np
import pandas as pd

for index in range(1, 10**6 + 1):
    # Total number of iterations
    n = 10**0 * index

    # Where are we
    print(n)

    # Number of oscillations
    p = 10**2**np.pi

    # Dumping ratio
    # a = p / (5*p + 1)**2

    # Dialectical functions

    def psi_1(x, s1, s2, i1, i2):
        """Being"""
        x = x.real
        k = x % np.pi
        real_part = np.cos(x) * s1
        imag_part = np.sin(x) * s2
        return real_part + 1j * imag_part

    def psi_2(x, s1, s2, i1, i2):
        """Nothing"""
        return psi_1(x, s1, s2, i1, i2) + psi_3(x, s1, s2, i1, i2)

    def psi_3(x, s1, s2, i1, i2):
        """Becoming — Difference — Identity"""
        x = x.real
        k = x % np.pi
        real_part = np.cos(x) * s1
        imag_part = -np.sin(x) * s2
        return real_part + 1j * imag_part

    def psi_4(x, s1, s2, i1, i2):
        """Passing"""
        return psi_2(psi_1(x, s1, s2, i1, i2), s1, s2, i1, i2)

    def psi_5(x, s1, s2, i1, i2):
        """Arising"""
        return psi_1(psi_2(x, s1, s2, i1, i2), s1, s2, i1, i2)

    def psi_6(x, s1, s2, i1, i2):
        """Equilibrium"""
        return psi_4(psi_3(x, s1, s2, i1, i2), s1, s2, i1, i2) - psi_5(
            psi_3(x, s1, s2, i1, i2), s1, s2, i1, i2
        )

    def psi_7(x, s1, s2, i1, i2):
        """Emerging"""
        return psi_3(psi_1(x, s1, s2, i1, i2), s1, s2, i1, i2)

    def psi_8(x, s1, s2, i1, i2):
        """Dissolving"""
        return psi_6(psi_2(x, s1, s2, i1, i2), s1, s2, i1, i2)

    def psi_9(x, s1, s2, i1, i2):
        """Harmony"""
        return psi_8(psi_3(x, s1, s2, i1, i2), s1, s2, i1, i2) - psi_7(
            psi_3(x, s1, s2, i1, i2), s1, s2, i1, i2
        )

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

    def psi(state, x, s1, s2, i1, i2):
        return psi_functions[state](x, s1, s2, i1, i2)

    # Generate x values
    x_vals = np.linspace(-p, p, n, dtype=np.complex128)

    # Generate spin values
    s_vals = np.random.choice([-1, 1], [2, n])

    # Generate inverse values
    i_vals = np.random.choice([-1, 1], [2, n])

    # Generate steps

    values = []

    state = None

    for i, x in enumerate(x_vals):
        if state is None:
            state = 0

        else:
            state = next_state(state)

        values.append(
            psi(state, x, s_vals[0][i], s_vals[1][i], i_vals[0][i], i_vals[1][i])
        )

    # Evaluate psi(x) for these x values

    psi_vals = np.array(values)

    # Normalise data

    data = np.vstack([x_vals.real, psi_vals.real, psi_vals.imag])

    normalised = (data - data.min()) / (data.max() - data.min())

    # Extract real and imaginary parts for mapping x -> x, y -> y.real, z -> y.imag

    df = pd.DataFrame(normalised.T, columns=("x", "Ψ(x)", "Ψ*(x)"))

    # Plotting

    # Save to disk

    # df.to_csv("data.v10.csv")

    df.to_hdf("data.h5", f"v10n{index}")
