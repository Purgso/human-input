from collections.abc import Sequence
from math import erf, log, log2, sqrt

import numpy as np
from numpy.typing import NDArray

from .speed import speed

_rng = np.random.default_rng()

Point = tuple[float, float]


def generate_trajectory(
    start: Point,
    end: Point | Sequence[Point],
    target_size: float | tuple[float, float],
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    def fitts_law(
        displacement: NDArray[np.float64], target_width: float, target_height: float
    ) -> float:
        distance = np.linalg.norm(displacement)
        if distance == 0:
            return speed.fitts_a
        effective_width = max(
            1.0,
            (abs(displacement[0]) * target_width + abs(displacement[1]) * target_height)
            / distance,
        )
        return speed.fitts_a + speed.fitts_b * log2(1 + distance / effective_width)

    def randomize_endpoint(
        endpoint: NDArray[np.float64], target_width: float, target_height: float
    ) -> NDArray[np.float64]:
        r = _rng.uniform(0, 1)
        theta = _rng.uniform(0, 2 * np.pi)
        dx = r * target_width * np.cos(theta) / 2.0
        dy = r * target_height * np.sin(theta) / 2.0
        return endpoint + np.array([dx, dy], dtype=np.float64)

    def generate_guidepoint(
        startpoint: NDArray[np.float64], endpoint: NDArray[np.float64]
    ) -> NDArray[np.float64] | None:
        displacement = endpoint - startpoint
        distance = float(np.linalg.norm(displacement))

        if distance == 0:
            return None

        direction = displacement / distance
        perpendicular = np.array([-direction[1], direction[0]], dtype=np.float64)

        bend_sigma = distance * speed.bend_fraction
        bend = float(_rng.normal(0.2 * bend_sigma, bend_sigma))
        return startpoint + displacement * _rng.uniform(0.6, 0.8) + perpendicular * bend

    def generate_waypoints(
        startpoint: NDArray[np.float64],
        endpoints: NDArray[np.float64],
        target_size: tuple[float, float],
        corrections: int,
    ) -> tuple[NDArray[np.float64], list[int]]:
        points = [startpoint]
        endpoint_indices = []

        for endpoint in endpoints:
            guidepoint = generate_guidepoint(points[-1], endpoint)
            if guidepoint is not None:
                points.append(guidepoint)
            points.append(endpoint)
            endpoint_indices.append(len(points) - 1)

        if corrections == 0:
            return np.asarray(points, dtype=np.float64), endpoint_indices

        # Replace the final endpoint with an initial miss followed by progressively
        # smaller corrections. Intermediate targets are approached directly.
        endpoint = endpoints[-1]
        segment_start = endpoints[-2] if len(endpoints) > 1 else startpoint
        displacement = endpoint - segment_start
        distance = float(np.linalg.norm(displacement))
        if distance == 0:
            return np.asarray(points, dtype=np.float64), endpoint_indices

        direction = displacement / distance
        perpendicular = np.array([-direction[1], direction[0]], dtype=np.float64)
        target_scale = sqrt(target_size[0] * target_size[1])

        points.pop()

        error_scale = min(target_scale * speed.initial_error_scale, distance * 0.15)
        longitudinal_error = _rng.normal(0.0, error_scale)
        lateral_error = _rng.normal(0.0, error_scale * 0.7)
        error = direction * longitudinal_error + perpendicular * lateral_error
        points.append(endpoint + error)

        remaining_error = error
        for correction_index in range(1, corrections):
            decay = _rng.normal(speed.correction_decay, 0.10)
            decay = float(np.clip(decay, 0.10, 0.65))

            if _rng.random() < 0.25:
                decay *= -1.0
            remaining_error *= decay

            correction_noise = _rng.normal(
                0.0, error_scale * 0.08 / correction_index, size=2
            )
            points.append(endpoint + remaining_error + correction_noise)

        points.append(endpoint)
        endpoint_indices[-1] = len(points) - 1

        return np.asarray(points, dtype=np.float64), endpoint_indices

    def catmull_rom_segments(
        points: NDArray[np.float64], samples_per_segment: int = 50
    ) -> list[NDArray[np.float64]]:
        if len(points) == 2:
            t = np.linspace(0.0, 1.0, samples_per_segment)
            return [
                points[0][None, :] * (1.0 - t)[:, None]
                + points[1][None, :] * t[:, None]
            ]

        padded = np.vstack([points[0], points, points[-1]])

        segments = []

        for i in range(1, len(padded) - 2):
            p0, p1, p2, p3 = padded[i - 1], padded[i], padded[i + 1], padded[i + 2]
            t = np.linspace(0.0, 1.0, samples_per_segment)
            t2 = t * t
            t3 = t2 * t
            segment = 0.5 * (
                (2 * p1)
                + (-p0 + p2) * t[:, None]
                + (2 * p0 - 5 * p1 + 4 * p2 - p3) * t2[:, None]
                + (-p0 + 3 * p1 - 3 * p2 + p3) * t3[:, None]
            )
            segments.append(segment)

        return segments

    def schedule_path(
        path: NDArray[np.float64], duration: float
    ) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        def lognormal_progress(timestamps: NDArray[np.float64]) -> NDArray[np.float64]:
            peak_time = max(duration * speed.velocity_peak, 1e-6)
            mu = log(peak_time) + speed.lognormal_sigma**2
            safe_time = np.maximum(timestamps, 1e-9)

            z = (np.log(safe_time) - mu) / (speed.lognormal_sigma * np.sqrt(2))

            cdf = np.asarray(
                [0.5 * (1.0 + erf(z_val)) for z_val in z], dtype=np.float64
            )
            cdf[0] = 0.0

            total = cdf[-1]
            if total <= 0:
                return np.linspace(0.0, 1.0, len(timestamps))

            return cdf / total

        def curvature_weighted_progress() -> NDArray[np.float64]:
            segment_vectors = np.diff(path, axis=0)
            segment_lengths = np.linalg.norm(segment_vectors, axis=1)

            safe_lengths = np.maximum(segment_lengths, 1e-9)
            directions = segment_vectors / safe_lengths[:, None]

            curvature = np.zeros(len(path), dtype=np.float64)

            if len(directions) > 1:
                dot_products = np.sum(directions[:-1] * directions[1:], axis=1)
                dot_products = np.clip(dot_products, -1.0, 1.0)

                curvature[1:-1] = np.arccos(dot_products)

            segment_curvatures = (curvature[:-1] + curvature[1:]) / 2.0
            weighted_lengths = segment_lengths * (
                1.0 + segment_curvatures * speed.curvature_slowdown
            )

            cumulative = np.concatenate([[0.0], np.cumsum(weighted_lengths)])

            if cumulative[-1] == 0:
                return np.linspace(0.0, 1.0, len(path))

            return cumulative / cumulative[-1]

        sample_count = max(2, round(duration * 60) + 1)

        timestamps = np.linspace(0.0, duration, sample_count)

        progress = lognormal_progress(timestamps)

        path_progress = curvature_weighted_progress()

        x = np.interp(progress, path_progress, path[:, 0])
        y = np.interp(progress, path_progress, path[:, 1])
        positions = np.column_stack([x, y])
        return timestamps, positions

    if isinstance(target_size, (int, float)):
        target_size = (target_size, target_size)
    target_width, target_height = target_size
    if target_width <= 0 or target_height <= 0:
        raise ValueError("Target size must be positive.")

    startpoint = np.asarray(start, dtype=np.float64)
    target_centers = np.asarray(end, dtype=np.float64)
    if target_centers.shape == (2,):
        target_centers = target_centers.reshape(1, 2)
    if target_centers.ndim != 2 or target_centers.shape[1] != 2:
        raise ValueError("Targets must be a point or a non-empty series of points.")
    if len(target_centers) == 0:
        raise ValueError("Targets must contain at least one point.")

    centers_with_start = np.vstack([startpoint, target_centers])
    move_times = np.asarray(
        [
            fitts_law(displacement, target_width, target_height)
            for displacement in np.diff(centers_with_start, axis=0)
        ],
        dtype=np.float64,
    )
    endpoints = np.asarray(
        [
            randomize_endpoint(center, target_width, target_height)
            for center in target_centers
        ],
        dtype=np.float64,
    )
    corrections = _rng.choice(
        len(speed.correction_probability), p=speed.correction_probability
    )
    waypoints, endpoint_indices = generate_waypoints(
        startpoint, endpoints, (target_width, target_height), corrections
    )
    path_segments = catmull_rom_segments(waypoints)

    timestamp_sections = []
    position_sections = []
    start_segment = 0
    elapsed = 0.0

    for duration, endpoint_index in zip(move_times, endpoint_indices, strict=True):
        section_path = np.vstack(path_segments[start_segment:endpoint_index])
        section_timestamps, section_positions = schedule_path(section_path, duration)

        if timestamp_sections:
            section_timestamps = section_timestamps[1:]
            section_positions = section_positions[1:]

        timestamp_sections.append(section_timestamps + elapsed)
        position_sections.append(section_positions)
        elapsed += duration
        start_segment = endpoint_index

    return np.concatenate(timestamp_sections), np.vstack(position_sections)
