import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt

mpl.rcParams["agg.path.chunksize"] = 10000


# Define the functions
def psi_1(x):
    return np.cos(x) + 1j * np.sin(x)


def psi_2(x):
    return -np.sin(x) + 1j * np.cos(x)


def psi_3(x):
    return psi_1(x) + psi_2(x)


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

# Define dependencies

dependencies = {
    1: [],
    2: [],
    3: [1, 2],
    4: [1, 2],
    5: [1, 2],
    6: [1, 2, 3, 4, 5],
    7: [1, 2, 3],
    8: [1, 2, 3, 4, 5, 6],
    9: [1, 2, 3, 4, 5, 6, 7, 8],
}


# Initialise transition matrix

n_states = len(psi_functions)
transition_matrix = np.zeros((n_states, n_states))

# Populate transition matrix with appropriate probabilities

for i in range(n_states):
    for j in range(n_states):
        if j not in dependencies[i + 1]:
            transition_matrix[i, j] = 1 / (n_states - len(dependencies[i + 1]))


# Function to simulate one step of the Markov chain


def next_state(current_state):
    return np.random.choice(range(9), p=transition_matrix[current_state])


# Define the composite function Ψ(x) with stochastic selection


def Psi(state, x):
    return psi_functions[state](x)


# Generate x values

x1_vals = np.linspace(0, 2 * np.pi, 100000, dtype="complex")
x2_vals = np.linspace(0, 2 * np.pi, 100000, dtype="complex")
x3_vals = np.linspace(0, 2 * np.pi, 100000, dtype="complex")

# Evaluate Psi(x) for these x values

values = []
state = None
dependencies[1] = []
dependencies[2] = [1]

for i, x in enumerate(x1_vals):
    if state is None:
        state = 0
    else:
        state = next_state(state)

    values = [*values, Psi(state, x)]


Psi1_vals = np.array(values)

values = []
state = None
dependencies[1] = [2]
dependencies[2] = []

for i, x in enumerate(x2_vals):
    if state is None:
        state = 1
    else:
        state = next_state(state)

    values = [*values, Psi(state, x)]

Psi2_vals = np.array(values)

values = []
state = None
dependencies[1] = []
dependencies[2] = []

for i, x in enumerate(x3_vals):
    if state is None:
        state = np.random.choice([0, 1])
    else:
        state = next_state(state)

    values = [*values, Psi(state, x)]

Psi3_vals = np.array(values)


# Extract real and imaginary parts for mapping x -> x, y -> y.real, z -> y.imag

x1_coord = x1_vals.real
y1_coord = Psi1_vals.real
z1_coord = Psi1_vals.imag
x2_coord = x2_vals.real
y2_coord = Psi2_vals.real
z2_coord = Psi2_vals.imag
x3_coord = x3_vals.real
y3_coord = Psi3_vals.real
z3_coord = Psi3_vals.imag

# Plotting

fig = plt.figure(15, figsize=(27, 10))

a1 = fig.add_subplot(1, 3, 1, projection="3d")
a1.plot(x1_coord, y1_coord, z1_coord, linewidth=0.015, label="Ψ(x)")
a1.set_title("Nothing depends on being")
a1.set_xlabel("x")
a1.set_ylabel("Ψ(x)")
a1.set_zlabel("iΨ(x)")
a1.legend()

a2 = fig.add_subplot(1, 3, 2, projection="3d")
a2.plot(x2_coord, y2_coord, z2_coord, linewidth=0.015, label="Ψ(x)")
a2.set_title("Being depends on nothing")
a2.set_xlabel("x")
a2.set_ylabel("Ψ(x)")
a2.set_zlabel("iΨ(x)")
a2.legend()

a3 = fig.add_subplot(1, 3, 3, projection="3d")
a3.plot(x3_coord, y3_coord, z3_coord, linewidth=0.015, label="Ψ(x)")
a3.set_title("Being and nothing are independent of each other")
a3.set_xlabel("x")
a3.set_ylabel("Ψ(x)")
a3.set_zlabel("iΨ(x)")
a3.legend()

plt.show()

plt.savefig("version6.png", dpi=2000, bbox_inches="tight")
