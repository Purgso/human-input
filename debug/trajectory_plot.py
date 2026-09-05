from human_input.trajectory import generate_trajectory

import numpy as np
import matplotlib.pyplot as plt


def plot_movement(
    path: np.ndarray,
    timestamps: np.ndarray,
    *,
    target_center: tuple[int, int] | None = None,
    target_size: tuple[int, int] | None = None,
) -> None:
    """
    Plot a 2-D mouse path and its speed profile.

    path: shape (n, 2), containing x/y positions
    timestamps: shape (n,), in seconds
    """
    velocity_x = np.gradient(path[:, 0], timestamps)
    velocity_y = np.gradient(path[:, 1], timestamps)
    speed = np.hypot(velocity_x, velocity_y)

    figure, (path_axis, speed_axis) = plt.subplots(
        1,
        2,
        figsize=(12, 5),
        layout="constrained",
    )

    # Path plot. Color shows elapsed time.
    points = path_axis.scatter(
        path[:, 0],
        path[:, 1],
        c=timestamps,
        cmap="viridis",
        s=12,
    )

    path_axis.plot(path[:, 0], path[:, 1], color="black", alpha=0.35)
    path_axis.scatter(*path[0], color="limegreen", s=80, label="Start", zorder=3)
    path_axis.scatter(*path[-1], color="red", s=80, label="End", zorder=3)

    if target_center is not None and target_size is not None:
        target_width, target_height = target_size
        target_x, target_y = target_center

        target = plt.Rectangle(
            (
                target_x - target_width / 2,
                target_y - target_height / 2,
            ),
            target_width,
            target_height,
            fill=False,
            edgecolor="tab:red",
            linewidth=2,
            label="Target",
        )
        path_axis.add_patch(target)

    figure.colorbar(points, ax=path_axis, label="Time (s)")

    path_axis.set_title("Mouse path")
    path_axis.set_xlabel("Screen X (px)")
    path_axis.set_ylabel("Screen Y (px)")
    path_axis.set_aspect("equal")
    path_axis.invert_yaxis()  # Match normal screen coordinates.
    path_axis.legend()

    speed_axis.plot(timestamps, speed, color="tab:blue")
    speed_axis.fill_between(timestamps, speed, color="tab:blue", alpha=0.2)

    speed_axis.set_title("Velocity profile")
    speed_axis.set_xlabel("Time (s)")
    speed_axis.set_ylabel("Speed (px/s)")
    speed_axis.set_xlim(timestamps[0], timestamps[-1])
    speed_axis.set_ylim(bottom=0)
    speed_axis.grid(alpha=0.3)

    plt.show()

start = (0, 0)
end = (200, 100)
target = (10, 20)

timestamps, path = generate_trajectory(start, end, target)

plot_movement(
    path,
    timestamps,
    target_center=end,
    target_size=target,
)