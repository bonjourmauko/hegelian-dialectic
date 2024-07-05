import aquarel
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib import animation

df = pd.read_hdf("data.h5", "v10")
dpi = 300
frames = 60
chunk_size = int(len(df["x"].T) / frames)


def update_lines(num, lines):
    chunk = num * chunk_size

    for i, line in zip(range(chunk), lines):
        line.set_data_3d([df["x"][:i], df["Ψ(x)"][:i], df["Ψ*(x)"][:i]])

    return lines


with aquarel.load_theme("arctic_dark"):
    mpl.rcParams["agg.path.chunksize"] = 10000

    plt.axis("off")

    fig = plt.figure(figsize=(13, 11))

    ax = fig.add_subplot(111, projection="3d", elev=108, azim=-49, roll=0)
    # ax = fig.add_subplot(111, projection="3d")
    ax.set_box_aspect(aspect=(10, 1, 1))
    ax.margins(tight=True)
    ax.set_axis_off()

    # lines = [
    #     ax.plot([], [], [], linewidth=1 / (i * 10 + 1))[0]
    #     for i in range(frames * chunk_size)
    # ]

    # anim = animation.FuncAnimation(
    #     fig, update_lines, frames, fargs=(lines,), interval=1
    # )

    ax.set_title("Existence")
    ax.plot(df["x"], df["Ψ(x)"], df["Ψ*(x)"], linewidth=0.001)
    plt.subplots_adjust(top=1.2, bottom=-0.2, right=1.6, left=-0.6)

    # writervideo = animation.FFMpegWriter(fps=frames / 2)

    # anim.save("version10-2.mp4", writer=writervideo)

    # plt.close()

    # plt.show()

    for i in range(1, 8):
        plt.savefig(f"version10-{i}.png", dpi=dpi * i, bbox_inches="tight")
