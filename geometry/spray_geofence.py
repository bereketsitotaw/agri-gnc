"""
Smallholder spray geofencing — decompose Boustrophedon legs by authorized plots.

Uses Shapely line–multipolygon intersection to tag spray ON/OFF segments along
each sweep without custom ray-casting.
"""

from __future__ import annotations

from dataclasses import dataclass

from shapely.geometry import LineString, MultiPolygon, Point, Polygon
from shapely.geometry.base import BaseGeometry
from shapely.ops import unary_union


@dataclass(frozen=True)
class SprayLegSlice:
    """Sub-segment of a spray leg with legally permitted pump state."""

    start: tuple[float, float]
    end: tuple[float, float]
    is_spraying: bool


class SprayGeofence:
    """Authorized application zones for participating smallholder plots."""

    def __init__(self, valid_polygons: list[Polygon]):
        if not valid_polygons:
            raise ValueError("SprayGeofence requires at least one valid polygon.")
        cleaned = [p for p in valid_polygons if not p.is_empty]
        if not cleaned:
            raise ValueError("SprayGeofence requires at least one non-empty polygon.")
        merged = unary_union(cleaned)
        if merged.geom_type == "Polygon":
            self.active_zone = MultiPolygon([merged])
        elif merged.geom_type == "MultiPolygon":
            self.active_zone = merged
        else:
            raise ValueError(f"Expected polygonal zones, got {merged.geom_type}")

    def slice_spray_leg(
        self,
        start_wp: tuple[float, float],
        end_wp: tuple[float, float],
        *,
        tol_m: float = 1e-6,
    ) -> list[SprayLegSlice]:
        """
        Split one Boustrophedon leg into ordered sub-segments tagged ON/OFF.

        Segments are listed from start_wp toward end_wp. Gaps outside
        ``active_zone`` are transit (spray OFF); overlaps are spray ON.
        """
        start = (float(start_wp[0]), float(start_wp[1]))
        end = (float(end_wp[0]), float(end_wp[1]))
        flight_path = LineString([start, end])
        total_len = flight_path.length

        if total_len <= tol_m:
            inside = self.active_zone.contains(Point(start))
            return [SprayLegSlice(start=start, end=end, is_spraying=inside)]

        on_intervals = self._spray_intervals_along_line(flight_path, tol_m)
        return self._build_slices_from_intervals(
            flight_path, start, end, total_len, on_intervals, tol_m
        )

    def _spray_intervals_along_line(
        self,
        flight_path: LineString,
        tol_m: float,
    ) -> list[tuple[float, float]]:
        """Return [start_dist, end_dist] intervals (ON) along the flight path."""
        overlap = flight_path.intersection(self.active_zone)
        parts = _line_parts(overlap)
        intervals: list[tuple[float, float]] = []

        for part in parts:
            if part.length <= tol_m:
                continue
            d0 = flight_path.project(Point(part.coords[0]), normalized=False)
            d1 = flight_path.project(Point(part.coords[-1]), normalized=False)
            lo, hi = (d0, d1) if d0 <= d1 else (d1, d0)
            if hi - lo > tol_m:
                intervals.append((lo, hi))

        if not intervals:
            return []

        intervals.sort(key=lambda iv: iv[0])
        merged: list[tuple[float, float]] = [intervals[0]]
        for lo, hi in intervals[1:]:
            prev_lo, prev_hi = merged[-1]
            if lo <= prev_hi + tol_m:
                merged[-1] = (prev_lo, max(prev_hi, hi))
            else:
                merged.append((lo, hi))
        return merged

    def _build_slices_from_intervals(
        self,
        flight_path: LineString,
        start: tuple[float, float],
        end: tuple[float, float],
        total_len: float,
        on_intervals: list[tuple[float, float]],
        tol_m: float,
    ) -> list[SprayLegSlice]:
        slices: list[SprayLegSlice] = []
        cursor = 0.0

        def point_at(dist: float) -> tuple[float, float]:
            pt = flight_path.interpolate(dist)
            return (float(pt.x), float(pt.y))

        for lo, hi in on_intervals:
            lo = max(0.0, lo)
            hi = min(total_len, hi)
            if hi <= lo + tol_m:
                continue
            if lo > cursor + tol_m:
                slices.append(
                    SprayLegSlice(
                        start=point_at(cursor),
                        end=point_at(lo),
                        is_spraying=False,
                    )
                )
            slices.append(
                SprayLegSlice(
                    start=point_at(lo),
                    end=point_at(hi),
                    is_spraying=True,
                )
            )
            cursor = hi

        if cursor < total_len - tol_m:
            slices.append(
                SprayLegSlice(
                    start=point_at(cursor),
                    end=end,
                    is_spraying=False,
                )
            )
        elif not slices:
            inside = self.active_zone.contains(Point(start))
            slices.append(
                SprayLegSlice(start=start, end=end, is_spraying=inside)
            )

        if slices:
            slices[0] = SprayLegSlice(start=start, end=slices[0].end, is_spraying=slices[0].is_spraying)
            slices[-1] = SprayLegSlice(
                start=slices[-1].start,
                end=end,
                is_spraying=slices[-1].is_spraying,
            )

        return _merge_adjacent_slices(slices, tol_m)


def _line_parts(geom: BaseGeometry | None) -> list[LineString]:
    if geom is None or geom.is_empty:
        return []
    if geom.geom_type == "LineString":
        return [geom]
    if geom.geom_type == "MultiLineString":
        return list(geom.geoms)
    if geom.geom_type == "GeometryCollection":
        parts: list[LineString] = []
        for g in geom.geoms:
            parts.extend(_line_parts(g))
        return parts
    return []


def _merge_adjacent_slices(
    slices: list[SprayLegSlice],
    tol_m: float,
) -> list[SprayLegSlice]:
    if not slices:
        return slices
    merged: list[SprayLegSlice] = [slices[0]]
    for seg in slices[1:]:
        prev = merged[-1]
        if (
            prev.is_spraying == seg.is_spraying
            and LineString([prev.end, seg.start]).length <= tol_m
        ):
            merged[-1] = SprayLegSlice(
                start=prev.start,
                end=seg.end,
                is_spraying=prev.is_spraying,
            )
        else:
            merged.append(seg)
    return merged
