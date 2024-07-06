import random
import pandas
import pyqtgraph.exporters

# Config
image_width = 4480
image_height = 2520
background_colors = [
    *[
        "#2e3440",
        "#3b4252",
        "#434c5e",
        "#4c566a",
        "#111115",
        "#2e2137",
        "#db904e",
        "#c74a00",
        "#770000",
        "#0e1820",
        "#0f334d",
        "#0b5d85",
    ],
    *["#000000"] * 100,
]
pen_colors = [
    *[
        "#d8dee9",
        "#e5e9f0",
        "#eceff4",
        "#f40234",
        "#f0532c",
        "#f0882f",
        "#688b2e",
        "#013f73",
        "#ffc7c7",
        "#ffe4e1",
        "#dcecf5",
        "#bad8eb",
        "#e9f7f7",
        "#edf8f8",
        "#f1fbfb",
        "#f4fbfb",
        "#f9ffff",
    ],
    *["#ffffff"] * 100,
]

data = pandas.read_hdf("data.h5", f"v10n{10**7}")

for index in range(1, 10001):
    # Read data
    # data = pandas.read_hdf("data.h5", f"v10n{index}")

    j = 10**2
    n = index * j

    # Set background
    background_color = pyqtgraph.mkColor(random.choice(background_colors))
    pyqtgraph.setConfigOption("background", background_color)
    pyqtgraph.setConfigOption("foreground", background_color)
    pyqtgraph.setConfigOption("useOpenGL", True)
    pyqtgraph.setConfigOption("useCupy", True)
    pyqtgraph.setConfigOption("useNumba", False)

    # Create line
    pen_color = pyqtgraph.mkColor(random.choice(pen_colors))
    pen = pyqtgraph.mkPen(color=pen_color, width=10.0 / (index / 10))

    # Plot
    plot = pyqtgraph.plot()
    plot.setGeometry(0, 0, image_width, image_height)
    plot.plot(data["Ψ(x)"][: (n - j)], data["Ψ*(x)"][: (n - j)], pen=pen)

    # Export
    exporter = pyqtgraph.exporters.ImageExporter(plot.plotItem)
    # exporter.params["width"] = image_width
    # exporter.params["height"] = image_height
    exporter.export(f"img/{index}.png")
