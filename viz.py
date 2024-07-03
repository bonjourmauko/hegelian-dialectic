import aquarel
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib import animation

df = pd.read_hdf("data.h5", "v8")


def update_lines(num, lines):
    for _, line in zip(df["x"], lines):
        line.set_data_3d([df["x"][:num], df["Ψ(x)"][:num], df["Ψ*(x)"][:num]])
        line.set_linewidth = 1 / (num + 1)

    return lines


with aquarel.load_theme("arctic_dark"):
    mpl.rcParams["agg.path.chunksize"] = 10000

    plt.axis("off")
    plt.subplots_adjust(top=1.5, bottom=-0.2, right=1.1, left=-0.3)

    fig = plt.figure(figsize=(14, 10))

    ax = fig.add_subplot(111, projection="3d", elev=15, azim=-15)
    ax.set_box_aspect(aspect=(2, 1, 1))
    ax.margins(tight=True)
    ax.set_axis_off()

    lines = [ax.plot([], [], [])[0] for _ in df["x"]]

    anim = animation.FuncAnimation(
        fig, update_lines, len(df["x"].T), fargs=(lines,), interval=100
    )

    plt.subplots_adjust(top=1.5, bottom=-0.2, right=1.1, left=-0.3)

    plt.show()
