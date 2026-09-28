import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt

stages = [
    "Preprocessing",
    "DPU inference",
    "Postprocessing",
    "HLS Sobel",
]

latency = [
    10.3329,
    7.24071,
    1.97276,
    1.74246,
]

fig, ax = plt.subplots(figsize=(10, 5))

bars = ax.barh(stages, latency)

ax.invert_yaxis()
ax.set_xlim(0, 13)
ax.set_xlabel("Average processing time (ms)")
ax.set_title("KV260 Live CARLA Vision Pipeline")

for bar, value in zip(bars, latency):
    ax.text(
        value + 0.15,
        bar.get_y() + bar.get_height() / 2,
        f"{value:.2f} ms",
        va="center",
    )

ax.grid(axis="x", alpha=0.25)
ax.set_axisbelow(True)

fig.text(
    0.12, 0.025,
    "696 frames | Compute: 46.97 FPS | "
    "End-to-end: 10.29 FPS | CARLA 640x480",
    fontsize=10,
)

fig.tight_layout(rect=[0, 0.08, 1, 1])

fig.savefig(
    "benchmarks/figures/live_pipeline_latency.png",
    dpi=200,
    bbox_inches="tight",
)

print("Performance figure generated successfully.")
