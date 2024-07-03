import random
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


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


iterations = 25000

# Generate x values from 0 to increasing multiples of pi
x_ranges = [
    np.linspace(0, 2 * np.pi, iterations, dtype="complex"),
    np.linspace(2 * np.pi, 4 * np.pi, iterations, dtype="complex"),
    np.linspace(4 * np.pi, 6 * np.pi, iterations, dtype="complex"),
    np.linspace(6 * np.pi, 10 * np.pi, iterations, dtype="complex"),
    np.linspace(10 * np.pi, 16 * np.pi, iterations, dtype="complex"),
    np.linspace(16 * np.pi, 26 * np.pi, iterations, dtype="complex"),
    np.linspace(26 * np.pi, 42 * np.pi, iterations, dtype="complex"),
    np.linspace(42 * np.pi, 68 * np.pi, iterations, dtype="complex"),
    np.linspace(68 * np.pi, 110 * np.pi, iterations, dtype="complex"),
]

# Calculate y values for each function
y_funcs = [psi_1, psi_2, psi_3, psi_4, psi_5, psi_6, psi_7, psi_8, psi_9]
y_values = [func(x) for func, x in zip(y_funcs, x_ranges)]

# Initialize empty lists for v values
v_values = [[] for _ in range(9)]

# Progressive entanglement using random selection
for i in range(iterations):
    for j in range(9):
        choices = [y_values[k][i] for k in range(j + 1)]
        v_values[j].append(random.choice(choices))

# Flatten v_values and concatenate x_values
y = np.concatenate(v_values)
x = np.concatenate(x_ranges)

# Plotting in 3D
fig = plt.figure(figsize=(18, 10))
ax = fig.add_subplot(111, projection="3d")
ax.set_box_aspect(aspect=(3, 1, 1))
ax.plot(x.real, y.real, y.imag, linewidth=0.1, label="f(x) = [x, rψ(x), iψ(x)]")
ax.set_xlabel("x = moments")
ax.set_ylabel("rψ(x) = metamorphosing (real)")
ax.set_zlabel("iψ(x) = metamorphosing (complex)")
ax.set_title("Vector-valued function f(x) over multiple intervals")
ax.legend()

plt.show()
