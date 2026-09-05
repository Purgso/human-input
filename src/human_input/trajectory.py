from collections.abc import Sequence
from math import erf, log, log2, sqrt

import random

import numpy as np
from numpy.typing import NDArray

from .speed import speed

_rng = random.Random()

Point = tuple[float, float]

"""
Create a human-like trajectory through two or more points.
1. Calculate the control points by randomly shifting the end points based on the size of the target.
2. Add 0-3 correction points before the final target point.
3. Generate a smooth trajectory through the control points.
4. Add curve points between control points at random offsets from the smooth trajectory.
5. Create a series of Bezier curves through the control and curve points to form the final trajectory.
6. Create lognormal velocity profiles for each segment of the trajectory.
7. Overlap the velocity profiles of consecutive segments to create a smooth overall motion that's slower near the control points.
8. Reduce the velocity profiles in regions of sharper curvature.
9. Generate the final path as a series of x and y coordinates with time stamps.
10. Add noise to the trajectory to simulate human-like imperfections.
11. Return the final trajectory
"""

def generate_trajectory(start: Point, end: Point | Sequence[Point], target_size: float | tuple[float, float]) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    def fitts_law(displacement: NDArray[np.float64]) -> float:
        distance = np.linalg.norm(displacement)
        if distance == 0:
            return speed.fitts_a
        effective_width = max(1.0, abs(displacement[0]) * target_width + abs(displacement[1]) * target_height) / distance
        return speed.fitts_a + speed.fitts_b * log2(distance / effective_width)

    def randomize_controlpoint(point: Point) -> Point:
        r = _rng.uniform(0, 0.95)
        theta = _rng.uniform(0, 2 * np.pi)
        dx = r * np.cos(theta) * target_width / 2.0
        dy = r * np.sin(theta) * target_height / 2.0
        return point[0] + dx, point[1] + dy

    def add_curve_points(points: list[Point]) -> None:
        curve_point_shifts = [_rng.uniform(0.35, 0.7) for _ in range(len(points) - 1)]
        distances = [sqrt((points[i+1][0] - points[i][0])**2 + (points[i+1][1] - points[i][1])**2) for i in range(len(points) - 1)]
        bend_sigmas = [d * speed.bend_fraction for d in distances]
        bends = [_rng.normalvariate(sigma, sigma) * _rng.choice([-1, 1]) for sigma in bend_sigmas]
        
        if len(points) == 2:
            displacement = (points[1][0] - points[0][0], points[1][1] - points[0][1])
            distance = sqrt(displacement[0]**2 + displacement[1]**2)
            direction = (displacement[0] / distance, displacement[1] / distance)
            perpendiculars = [(-direction[1], direction[0])]
            base_points = [(
                points[0][0] + curve_point_shifts[0] * (points[1][0] - points[0][0]),
                points[0][1] + curve_point_shifts[0] * (points[1][1] - points[0][1])
            )]
            
        else:
            padded = [points[0]] + points.copy() + [points[-1]]
            base_points = []
            perpendiculars = []
            for i in range(1, len(padded) - 2):
                p0, p1, p2, p3 = padded[i - 1], padded[i], padded[i + 1], padded[i + 2]
                t = curve_point_shifts[i - 1]
                t2 = t * t
                t3 = t2 * t
                x = 0.5 * ((2 * p1[0]) +
                        (-p0[0] + p2[0]) * t +
                        (2*p0[0] - 5*p1[0] + 4*p2[0] - p3[0]) * t2 +
                        (-p0[0] + 3*p1[0] - 3*p2[0] + p3[0]) * t3)
                y = 0.5 * ((2 * p1[1]) +
                        (-p0[1] + p2[1]) * t +
                        (2*p0[1] - 5*p1[1] + 4*p2[1] - p3[1]) * t2 +
                        (-p0[1] + 3*p1[1] - 3*p2[1] + p3[1]) * t3)
                base_points.append((x, y))

                ax = -p0[0] + p2[0]
                ay = -p0[1] + p2[1]
                bx = 2*p0[0] - 5*p1[0] + 4*p2[0] - p3[0]
                by = 2*p0[1] - 5*p1[1] + 4*p2[1] - p3[1]
                cx = -p0[0] + 3*p1[0] - 3*p2[0] + p3[0]
                cy = -p0[1] + 3*p1[1] - 3*p2[1] + p3[1]

                tangent_x = 0.5 * (ax + 2 * bx * t + 3 * cx * t2)
                tangent_y = 0.5 * (ay + 2 * by * t + 3 * cy * t2)

                tangent_length = sqrt(tangent_x**2 + tangent_y**2)
                if tangent_length == 0:
                    # Degenerate spline: fall back to the segment direction.
                    tangent_x = p2[0] - p1[0]
                    tangent_y = p2[1] - p1[1]
                    tangent_length = sqrt(tangent_x**2 + tangent_y**2)

                if tangent_length == 0:
                    perpendiculars.append((0.0, 0.0))
                else:
                    perpendiculars.append((
                        -tangent_y / tangent_length,
                        tangent_x / tangent_length,
                    ))

        curve_points = [(base_points[i][0] + bends[i] * perpendiculars[i][0], base_points[i][1] + bends[i] * perpendiculars[i][1]) for i in range(len(base_points))]
        for i in range(len(curve_points)-1, -1, -1):
            points.insert(i, curve_points[i])

    
    if isinstance(target_size, (int, float)):
        target_size = (target_size, target_size)
    target_width, target_height = target_size
    if target_width <= 0 or target_height <= 0:
        raise ValueError("Target size must be positive.")

    targets = [start]
    if isinstance(end, Sequence):
        targets.extend(end)
    else:
        targets.append(end)

    control_points = [randomize_controlpoint(point) for point in targets]