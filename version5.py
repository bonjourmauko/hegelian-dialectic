import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import random

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
psi_list = [psi_1, psi_2, psi_3, psi_4, psi_5, psi_6, psi_7, psi_8, psi_9]


# Define the composite function Ψ(x) with stochastic selection
def Psi(x):
    n = int(x // (2 * np.pi))
    if 0 <= n < 1:
        return psi_1(x)
    elif 1 <= n < 2:
        return random.choice([psi_1(x), psi_2(x)])
    elif 2 <= n < 3:
        return random.choice([psi_1(x), psi_2(x), psi_3(x)])
    elif 3 <= n < 4:
        return random.choice([psi_1(x), psi_2(x), psi_3(x), psi_4(x)])
    elif 4 <= n < 5:
        return random.choice([psi_1(x), psi_2(x), psi_3(x), psi_4(x), psi_5(x)])
    elif 5 <= n < 6:
        return random.choice(
            [psi_1(x), psi_2(x), psi_3(x), psi_4(x), psi_5(x), psi_6(x)]
        )
    elif 6 <= n < 7:
        return random.choice(
            [psi_1(x), psi_2(x), psi_3(x), psi_4(x), psi_5(x), psi_6(x), psi_7(x)]
        )
    elif 7 <= n < 8:
        return random.choice(
            [
                psi_1(x),
                psi_2(x),
                psi_3(x),
                psi_4(x),
                psi_5(x),
                psi_6(x),
                psi_7(x),
                psi_8(x),
            ]
        )
    elif 8 <= n < 9:
        return random.choice(
            [
                psi_1(x),
                psi_2(x),
                psi_3(x),
                psi_4(x),
                psi_5(x),
                psi_6(x),
                psi_7(x),
                psi_8(x),
                psi_9(x),
            ]
        )
    else:
        return psi_1(x)


# Generate x values
x_vals = np.linspace(0, 20 * np.pi, 100000000)
# Evaluate Psi(x) for these x values
Psi_vals = np.array([Psi(x) for x in x_vals])

# Extract real and imaginary parts for mapping x -> x, y -> y.real, z -> y.imag
x_coord = x_vals
y_coord = Psi_vals.real
z_coord = Psi_vals.imag

# Plotting
fig = plt.figure(figsize=(18, 10))
ax = fig.add_subplot(111, projection="3d")
ax.set_box_aspect(aspect=(3, 1, 1))
ax.plot(x_coord, y_coord, z_coord, linewidth=0.01, label="Ψ(x)")
ax.set_title("Progressive entanglement of Ψ(x)")
ax.set_xlabel("x")
ax.set_ylabel("Ψ(x)")
ax.set_zlabel("iΨ(x)")
ax.legend()

# plt.show()

plt.savefig("version5.png", dpi=3600, bbox_inches="tight")
