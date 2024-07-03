import random
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


# Define the functions (assuming they are already defined as in previous interactions)
def psi_1(x):
    return 1j * np.sin(x) + np.cos(x)


def psi_2(x):
    return 1j * np.cos(x) - np.sin(x)


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


def psi_7(x):
    return psi_3(psi_1(x))


def psi_8(x):
    return psi_6(psi_2(x))


def psi_9(x):
    return psi_7(psi_3(x)) + psi_8(psi_3(x))


iterations = 25000

# Generate x values from -pi to pi
x1 = np.linspace(0, 2 * np.pi, iterations, dtype="complex")
x2 = np.linspace(2 * np.pi, 4 * np.pi, iterations, dtype="complex")
x3 = np.linspace(4 * np.pi, 6 * np.pi, iterations, dtype="complex")
x4 = np.linspace(6 * np.pi, 10 * np.pi, iterations, dtype="complex")
x5 = np.linspace(10 * np.pi, 16 * np.pi, iterations, dtype="complex")
x6 = np.linspace(16 * np.pi, 26 * np.pi, iterations, dtype="complex")
x7 = np.linspace(26 * np.pi, 42 * np.pi, iterations, dtype="complex")
x8 = np.linspace(42 * np.pi, 68 * np.pi, iterations, dtype="complex")
x9 = np.linspace(68 * np.pi, 110 * np.pi, iterations, dtype="complex")

# Calculate y values for [t, rψ(t), iψ(t)]
y1 = psi_1(x1)
y2 = psi_2(x2)
y3 = psi_3(x3)
y4 = psi_4(x4)
y5 = psi_5(x5)
y6 = psi_6(x6)
y7 = psi_7(x7)
y8 = psi_8(x8)
y9 = psi_9(x9)

v1 = []
v2 = []
v3 = []
v4 = []
v5 = []
v6 = []
v7 = []
v8 = []
v9 = []

for i in range(0, iterations):
    v1 = [*v1, y1[i]]
    v2 = [*v2, random.choice([y1[i], y2[i]])]
    v3 = [*v3, random.choice([y1[i], y2[i], y3[i]])]
    v4 = [*v4, random.choice([y1[i], y2[i], y3[i], y4[i]])]
    v5 = [*v5, random.choice([y1[i], y2[i], y3[i], y4[i], y5[i]])]
    v6 = [*v6, random.choice([y1[i], y2[i], y3[i], y4[i], y5[i], y6[i]])]
    v7 = [*v7, random.choice([y1[i], y2[i], y3[i], y4[i], y5[i], y6[i], y7[i]])]
    v8 = [*v8, random.choice([y1[i], y2[i], y3[i], y4[i], y5[i], y6[i], y7[i], y8[i]])]
    v9 = [
        *v9,
        random.choice([y1[i], y2[i], y3[i], y4[i], y5[i], y6[i], y7[i], y8[i], y9[i]]),
    ]

y = np.array([*v1, *v2, *v3, *v4, *v5, *v6, *v7, *v8, *v9])
x = np.array([*x1, *x2, *x3, *x4, *x5, *x6, *x7, *x8, *x9])

# Plotting in 3D
fig = plt.figure(figsize=(18, 10))
ax = fig.add_subplot(111, projection="3d")
ax.set_box_aspect(aspect=(3, 1, 1))
ax.plot(x.real, y.real, y.imag, linewidth=0.1, label="f(x) = [x, rψ(x), iψ(x)]")
ax.set_xlabel("x = moments")
ax.set_ylabel("rψ(x) = metamorphosing (real)")
ax.set_zlabel("iψ(x) = metamorphosing (complex)")
ax.set_title("Vector-valued function f(x) over [-π, π]")
ax.legend()

plt.show()
