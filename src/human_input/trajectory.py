from __future__ import annotations

import random
from math import acos, cos, erf, log, log2, pi, sin, sqrt
from typing import Sequence, Tuple

from . import settings

_rng = random.Random()

Point = Tuple[float, float]


def generate_trajectory(
    start: Point,
    end: Point | Sequence[Point],
    target_size: float | tuple[float, float],
) -> tuple[list[float], list[float], list[float]]:
    def fitts_law(displacement: tuple[float, float]) -> float:
        distance = sqrt(displacement[0] ** 2 + displacement[1] ** 2)
        if distance == 0:
            return settings.settings.fitts_a
        effective_width = (
            max(
                1.0,
                abs(displacement[0]) * target_width
                + abs(displacement[1]) * target_height,
            )
            / distance
        )
        duration = settings.settings.fitts_a + settings.settings.fitts_b * log2(
            1.0 + distance / effective_width
        )
        return (
            min(
                settings.settings.maximum_time,
                max(settings.settings.minimum_time, duration),
            )
            / settings.speed.mouse_move_scaling
        )

    def randomize_controlpoint(point: Point) -> Point:
        r = _rng.uniform(0, 0.95)
        theta = _rng.uniform(0, 2 * pi)
        dx = r * cos(theta) * target_width / 2.0
        dy = r * sin(theta) * target_height / 2.0
        return point[0] + dx, point[1] + dy

    def turn_angles(points: list[Point]) -> list[float]:
        angles = []
        for i in range(1, len(points) - 1):
            a = points[i - 1]
            b = points[i]
            c = points[i + 1]
            ab = (b[0] - a[0], b[1] - a[1])
            bc = (c[0] - b[0], c[1] - b[1])
            dot_product = ab[0] * bc[0] + ab[1] * bc[1]
            mag_ab = sqrt(ab[0] ** 2 + ab[1] ** 2)
            mag_bc = sqrt(bc[0] ** 2 + bc[1] ** 2)
            if mag_ab == 0 or mag_bc == 0:
                angles.append(0.0)
            else:
                cos_angle = max(-1.0, min(1.0, dot_product / (mag_ab * mag_bc)))
                angles.append(acos(cos_angle))
        return angles

    def add_correction_points(points: list[Point]) -> None:
        num_corrections = _rng.choices(
            list(range(len(settings.settings.correction_probability))),
            weights=settings.settings.correction_probability,
            k=1,
        )[0]
        endpoint = points[-1]
        startpoint = points[-2]

        target_scale = sqrt(target_width * target_height)
        displacement = (endpoint[0] - startpoint[0], endpoint[1] - startpoint[1])
        distance = sqrt(displacement[0] ** 2 + displacement[1] ** 2)
        if distance == 0:
            return

        direction = (displacement[0] / distance, displacement[1] / distance)
        perpendicular = (-direction[1], direction[0])

        error_scale = min(
            target_scale * settings.settings.initial_error_scale, distance * 0.15
        )
        longitudinal_error = _rng.normalvariate(0.0, error_scale)
        lateral_error = _rng.normalvariate(0.0, error_scale * 0.7)
        error = (
            longitudinal_error * direction[0] + lateral_error * perpendicular[0],
            longitudinal_error * direction[1] + lateral_error * perpendicular[1],
        )
        noise = (0, 0)

        for correction_index in range(1, num_corrections + 1):
            correction_point = (
                endpoint[0] + error[0] + noise[0],
                endpoint[1] + error[1] + noise[1],
            )

            points.insert(-1, correction_point)

            decay = _rng.normalvariate(
                settings.settings.correction_decay,
                settings.settings.correction_decay / 3,
            )
            decay = min(0.65, max(0.10, decay))
            if _rng.random() < 0.25:
                decay *= -1
            error = (error[0] * decay, error[1] * decay)

            noise_sigma = error_scale * 0.08 / correction_index
            noise = (
                _rng.normalvariate(0.0, noise_sigma),
                _rng.normalvariate(0.0, noise_sigma),
            )

    def add_curve_points(points: list[Point]) -> None:
        curve_point_shifts = [_rng.uniform(0.35, 0.7) for _ in range(len(points) - 1)]
        distances = [
            sqrt(
                (points[i + 1][0] - points[i][0]) ** 2
                + (points[i + 1][1] - points[i][1]) ** 2
            )
            for i in range(len(points) - 1)
        ]
        bend_sigmas = [d * settings.settings.bend_fraction for d in distances]
        bends = [
            _rng.normalvariate(sigma, sigma) * _rng.choice([-1, 1])
            for sigma in bend_sigmas
        ]

        if len(points) == 2:
            displacement = (points[1][0] - points[0][0], points[1][1] - points[0][1])
            distance = sqrt(displacement[0] ** 2 + displacement[1] ** 2)
            if distance == 0:
                perpendiculars = [(0.0, 0.0)]
            else:
                direction = (displacement[0] / distance, displacement[1] / distance)
                perpendiculars = [(-direction[1], direction[0])]
            base_points = [
                (
                    points[0][0]
                    + curve_point_shifts[0] * (points[1][0] - points[0][0]),
                    points[0][1]
                    + curve_point_shifts[0] * (points[1][1] - points[0][1]),
                )
            ]

        else:
            padded = [points[0]] + points.copy() + [points[-1]]
            base_points = []
            perpendiculars = []
            for i in range(1, len(padded) - 2):
                p0, p1, p2, p3 = padded[i - 1], padded[i], padded[i + 1], padded[i + 2]
                t = curve_point_shifts[i - 1]
                t2 = t * t
                t3 = t2 * t
                tangent_scale = 0.5 * (1.0 - settings.settings.tension)

                m1x = tangent_scale * (p2[0] - p0[0])
                m1y = tangent_scale * (p2[1] - p0[1])
                m2x = tangent_scale * (p3[0] - p1[0])
                m2y = tangent_scale * (p3[1] - p1[1])

                h00 = 2 * t3 - 3 * t2 + 1
                h10 = t3 - 2 * t2 + t
                h01 = -2 * t3 + 3 * t2
                h11 = t3 - t2

                x = h00 * p1[0] + h10 * m1x + h01 * p2[0] + h11 * m2x
                y = h00 * p1[1] + h10 * m1y + h01 * p2[1] + h11 * m2y

                base_points.append((x, y))

                h00_derivative = 6 * t2 - 6 * t
                h10_derivative = 3 * t2 - 4 * t + 1
                h01_derivative = -6 * t2 + 6 * t
                h11_derivative = 3 * t2 - 2 * t

                tangent_x = (
                    h00_derivative * p1[0]
                    + h10_derivative * m1x
                    + h01_derivative * p2[0]
                    + h11_derivative * m2x
                )
                tangent_y = (
                    h00_derivative * p1[1]
                    + h10_derivative * m1y
                    + h01_derivative * p2[1]
                    + h11_derivative * m2y
                )

                tangent_length = sqrt(tangent_x**2 + tangent_y**2)
                if tangent_length == 0:
                    # Degenerate spline: fall back to the segment direction.
                    tangent_x = p2[0] - p1[0]
                    tangent_y = p2[1] - p1[1]
                    tangent_length = sqrt(tangent_x**2 + tangent_y**2)

                if tangent_length == 0:
                    perpendiculars.append((0.0, 0.0))
                else:
                    perpendiculars.append(
                        (
                            -tangent_y / tangent_length,
                            tangent_x / tangent_length,
                        )
                    )

        curve_points = [
            (
                base_points[i][0] + bends[i] * perpendiculars[i][0],
                base_points[i][1] + bends[i] * perpendiculars[i][1],
            )
            for i in range(len(base_points))
        ]
        for i, curve_point in enumerate(curve_points):
            points.insert(2 * i + 1, curve_point)

    def cubic_hermite_splines(
        points: list[Point],
        spline_count: int,
        samples_per_segment: int = 50,
    ) -> list[list[Point]]:
        padded = [points[0]] + points + [points[-1]]
        segments = []

        for i in range(1, len(padded) - 2):
            p0, p1, p2, p3 = padded[i - 1], padded[i], padded[i + 1], padded[i + 2]
            segment = []
            for j in range(samples_per_segment + 1):
                t = j / samples_per_segment
                t2 = t * t
                t3 = t2 * t
                tangent_scale = 0.5 * (1.0 - settings.settings.tension)

                m1x = tangent_scale * (p2[0] - p0[0])
                m1y = tangent_scale * (p2[1] - p0[1])
                m2x = tangent_scale * (p3[0] - p1[0])
                m2y = tangent_scale * (p3[1] - p1[1])

                h00 = 2 * t3 - 3 * t2 + 1
                h10 = t3 - 2 * t2 + t
                h01 = -2 * t3 + 3 * t2
                h11 = t3 - t2

                x = h00 * p1[0] + h10 * m1x + h01 * p2[0] + h11 * m2x
                y = h00 * p1[1] + h10 * m1y + h01 * p2[1] + h11 * m2y
                segment.append((x, y))
            segments.append(segment)

        splines = []
        segment_index = 0
        for spline_index in range(spline_count):
            if spline_index < spline_count - 1:
                segment_end = segment_index + 2
            else:
                # Correction points are inserted only before the final target, so
                # all remaining curve pieces belong to the final timed spline.
                segment_end = len(segments)

            spline = []
            for segment in segments[segment_index:segment_end]:
                spline.extend(segment if not spline else segment[1:])
            splines.append(spline)
            segment_index = segment_end

        return splines

    def generate_cdfs(steps: list[int]) -> list[list[float]]:
        cdfs = []
        sigma = settings.settings.lognormal_sigma
        peak_time = max(settings.settings.velocity_peak, 1e-6)
        mu = log(peak_time) + sigma**2

        for count in steps:
            cdf = [0.0]
            for index in range(1, count):
                timestamp = index / (count - 1)
                z = (log(timestamp) - mu) / (sigma * sqrt(2))
                cdf.append(0.5 * (1.0 + erf(z)))

            total = cdf[-1]
            if total <= 0:
                cdf = [index / (count - 1) for index in range(count)]
            else:
                cdf = [progress / total for progress in cdf]
                cdf[-1] = 1.0

            cdfs.append(cdf)

        return cdfs

    def progress_at_time(
        timestamp: float,
        start_time: float,
        duration: float,
        cdf: list[float],
    ) -> float:
        elapsed = timestamp - start_time
        if elapsed <= 0:
            return 0.0
        if elapsed >= duration:
            return 1.0

        cdf_position = elapsed / duration * (len(cdf) - 1)
        lower_index = int(cdf_position)
        fraction = cdf_position - lower_index
        return cdf[lower_index] * (1.0 - fraction) + cdf[lower_index + 1] * fraction

    if isinstance(target_size, (int, float)):
        target_size = (target_size, target_size)
    target_width, target_height = target_size
    if target_width <= 0 or target_height <= 0:
        raise ValueError("Target size must be positive.")

    if not isinstance(end, Sequence) or isinstance(end, (str, bytes)) or not end:
        raise ValueError("End must be a point or a non-empty series of points.")

    targets = [start]
    if not isinstance(end[0], (int, float)):
        targets.extend(end)  # type: ignore[arg-type]
    else:
        targets.append(end)  # type: ignore[arg-type]

    control_points = [start]
    control_points.extend(randomize_controlpoint(point) for point in targets[1:])
    add_correction_points(control_points)
    angles = turn_angles(control_points)

    durations = [
        fitts_law(
            (
                control_points[i][0] - control_points[i - 1][0],
                control_points[i][1] - control_points[i - 1][1],
            )
        )
        for i in range(1, len(control_points))
    ]
    steps = [max(2, round(duration * 120) + 1) for duration in durations]
    cdfs = generate_cdfs(steps)

    start_times = [0.0]
    for i in range(1, len(durations)):
        overlap = (
            (durations[i - 1] + durations[i])
            * settings.settings.impulse_overlap
            * (1.0 - settings.settings.curvature_slowdown * angles[i - 1] / pi)
        )
        start_times.append(start_times[-1] + durations[i - 1] - overlap)

    add_curve_points(control_points)
    segments = cubic_hermite_splines(control_points, len(cdfs))

    total_duration = start_times[-1] + durations[-1]

    output_steps = max(2, round(total_duration * 120) + 1)
    timestamps = [
        index / (output_steps - 1) * total_duration for index in range(output_steps)
    ]

    completion = [
        sum(
            progress_at_time(timestamp, start_time, duration, cdf)
            for start_time, duration, cdf in zip(start_times, durations, cdfs)
        )
        for timestamp in timestamps
    ]

    x = []
    y = []
    for progress in completion:
        if progress >= len(segments):
            point = segments[-1][-1]
        else:
            spline_index = int(progress)
            spline_progress = progress - spline_index
            spline = segments[spline_index]
            spline_position = spline_progress * (len(spline) - 1)
            lower_index = int(spline_position)
            fraction = spline_position - lower_index

            if lower_index == len(spline) - 1:
                point = spline[-1]
            else:
                point = (
                    spline[lower_index][0] * (1.0 - fraction)
                    + spline[lower_index + 1][0] * fraction,
                    spline[lower_index][1] * (1.0 - fraction)
                    + spline[lower_index + 1][1] * fraction,
                )

        x.append(point[0])
        y.append(point[1])

    return x, y, timestamps
