import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt

mpl.rcParams["agg.path.chunksize"] = 10000


# Define the functions


def psi_1(x):
    return np.cos(x) + 1j * np.sin(x)


def psi_2(x):
    return np.cos(x) - 1j * np.sin(x)


def psi_3(x):
    return psi_1(x) * psi_2(x)


def psi_4(x):
    return psi_2(psi_1(x))


def psi_5(x):
    return psi_1(psi_2(x))


def psi_6(x):
    return psi_4(psi_3(x)) + psi_5(psi_3(x))


def psi_7(x):
    return psi_3(psi_1(x))


def psi_8(x):
    return psi_6(psi_2(x))


def psi_9(x):
    return psi_7(psi_3(x)) + psi_8(psi_3(x))


# List of functions

psi_functions = [psi_1, psi_2, psi_3, psi_4, psi_5, psi_6, psi_7, psi_8, psi_9]

# Define the transition matrix (9x9) with equal probabilities

transition_matrix = np.full((9, 9), 1 / 9)


# Function to simulate one step of the Markov chain


def next_state(current_state):
    return np.random.choice(range(9), p=transition_matrix[current_state])


# Define the composite function Ψ(x) with stochastic selection


def Psi(state, x):
    return psi_functions[state](x)


# Generate x values

x_vals = np.linspace(0, 2 * np.pi, 500000, dtype="complex")


# Generate steps

values = []
state = 0

for i, x in enumerate(x_vals):
    if i != 0:
        state = next_state(state)

    values = [*values, Psi(state, x)]

# Evaluate Psi(x) for these x values

Psi_vals = np.array(values)

# Extract real and imaginary parts for mapping x -> x, y -> y.real, z -> y.imag

x_coord = x_vals.real
y_coord = Psi_vals.real
z_coord = Psi_vals.imag

# Plotting

fig = plt.figure(figsize=(18, 10))
ax = fig.add_subplot(111, projection="3d")
ax.set_box_aspect(aspect=(3, 1, 1))
ax.plot(x_coord, y_coord, z_coord, linewidth=0.001, label="Ψ(x)")
ax.set_title("Progressive entanglement of Ψ(x)")
ax.set_xlabel("x")
ax.set_ylabel("Ψ(x)")
ax.set_zlabel("iΨ(x)")
ax.legend()

# splt.show()

plt.savefig("version6.png", dpi=3600, bbox_inches="tight")
