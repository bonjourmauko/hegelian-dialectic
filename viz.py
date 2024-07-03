import aquarel
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt

df = pd.read_hdf("data.h5", "v8")


with aquarel.load_theme("arctic_dark"):
    mpl.rcParams["agg.path.chunksize"] = 10000

    plt.axis("off")

    fig = plt.figure(figsize=(14, 10))

    ax = fig.add_subplot(111, projection="3d", elev=15, azim=-15)
    ax.set_box_aspect(aspect=(2, 1, 1))
    ax.margins(tight=True)
    ax.plot(df["x"], df["Ψ(x)"], df["Ψ*(x)"], linewidth=0.008)
    ax.set_axis_off()

    plt.subplots_adjust(top=1.5, bottom=-0.2, right=1.1, left=-0.3)

    plt.show()
