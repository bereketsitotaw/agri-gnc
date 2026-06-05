from shapely.geometry import LineString, Point


def compute_path_metrics(segments: list[LineString]) -> dict:
    """
    Computes turn count and total path length for an ordered list of sweep segments.
    Total length = sum of segment lengths + sum of turnaround distances between segments.

    WARNING: Assumes projected metric (e.g., UTM) coordinates.
    Do not pass WGS84 coordinates, or Euclidean turnaround distances will be invalid.
    """
    if not segments:
        return {"turn_count": 0, "total_length": 0.0}

    turn_count = len(segments) - 1
    total_length = 0.0

    # Add lengths of the sweeps themselves
    for seg in segments:
        total_length += seg.length

    # Add turnaround distances (Euclidean distance from end of one sweep to start of next)
    for i in range(len(segments) - 1):
        end_point = Point(segments[i].coords[-1])
        start_next = Point(segments[i + 1].coords[0])
        total_length += end_point.distance(start_next)

    return {
        "turn_count": turn_count,
        "total_length": total_length
    }
