"""
Provides:
  1. A* shortest-path distance on map_agent.png
  2. Feature extractor: raw context -> fixed-length numpy vector

HOW TO USE:
    from features import get_features, set_env_bounds

    # Call ONCE after creating the environment
    set_env_bounds(env)

    # Call every time step
    feature_vector = get_features(context)
"""

import numpy as np
from PIL import Image
import heapq

# ---------------------------------------------------------------------------
# 1. Load map
#    White pixels (>=128) = road (walkable)
#    Black pixels (<128)  = building (blocked)
# ---------------------------------------------------------------------------

_map_image  = Image.open("map_agent.png").convert("L")
_map_array  = np.array(_map_image)
MAP_HEIGHT, MAP_WIDTH = _map_array.shape
ROAD = _map_array >= 128

# ---------------------------------------------------------------------------
# 2. Environment bounds
#    Only uses env.observation_space 
#    DELTA = size of one pixel in normalised units = max_x / MAP_WIDTH
# ---------------------------------------------------------------------------

_DELTA       = None
_MAX_DRIVERS = None
_MAX_X       = None
_MAX_Y       = None


def set_env_bounds(env):
    """
    Read coordinate bounds from env.observation_space.
    Must be called once before using get_features() or astar_distance().

    Parameters
    ----------
    env : DynamicPricingEnv instance
    """
    global _DELTA, _MAX_X, _MAX_Y, _MAX_DRIVERS
    passenger_high = env.observation_space[0].high
    _MAX_X = float(passenger_high[0])
    _MAX_Y = float(passenger_high[1])
    _DELTA       = _MAX_X / MAP_WIDTH
    _MAX_DRIVERS = float(env.MaxDrivers)


def _check_bounds():
    if _DELTA is None:
        raise RuntimeError("Call set_env_bounds(env) once before using features.")


# ---------------------------------------------------------------------------
# 3. Coordinate conversion
#    Observation gives normalised floats: x = col * delta, y = row * delta
#    We convert back to integer (row, col) for map lookups.
# ---------------------------------------------------------------------------

def _to_pixel(x, y):
    col = int(round(float(x) / _DELTA))
    row = int(round(float(y) / _DELTA))
    col = max(0, min(col, MAP_WIDTH  - 1))
    row = max(0, min(row, MAP_HEIGHT - 1))
    return row, col


def _nearest_road(row, col):
    """
    If (row, col) is a building pixel, BFS outward to find the nearest road.
    Handles coordinates that land just inside a building boundary.
    """
    if ROAD[row, col]:
        return row, col
    from collections import deque
    visited = np.zeros((MAP_HEIGHT, MAP_WIDTH), dtype=bool)
    visited[row, col] = True
    queue = deque([(row, col)])
    while queue:
        r, c = queue.popleft()
        for dr, dc in [(-1,0),(1,0),(0,-1),(0,1)]:
            nr, nc = r+dr, c+dc
            if 0 <= nr < MAP_HEIGHT and 0 <= nc < MAP_WIDTH and not visited[nr, nc]:
                if ROAD[nr, nc]:
                    return nr, nc
                visited[nr, nc] = True
                queue.append((nr, nc))
    return row, col


# ---------------------------------------------------------------------------
# 4. A* shortest-path
#
# Inspired by shortest_path() in RideSharing.py -- we use the same algorithm
# choice (A*) and the same 4-directional (Manhattan-style) movement on the
# road pixels of map_agent.png.
#
# Heuristic: plain Manhattan distance to destination.
#   h(r, c) = |r - dst_r| + |c - dst_c|
# This is admissible (never overestimates the true road distance) so A*
# is guaranteed to find the optimal path.
#
# Pseudocode
# ----------
# 1. Convert normalised (x,y) to pixel (row, col)
# 2. Snap any building pixels to nearest road pixel
# 3. Initialise: g(src) = 0, push (f = h(src), src) to priority queue
# 4. While queue not empty:
#      a. Pop pixel with smallest f = g + h
#      b. If pixel == dst: return g(pixel) * delta
#      c. For each 4-connected road neighbour:
#           tentative_g = g(current) + 1
#           if tentative_g < known g for neighbour:
#               g(neighbour) = tentative_g
#               push (tentative_g + h(neighbour), neighbour)
# 5. If queue empties: return map diagonal as penalty
#
# Caching
# -------
# Results stored in _astar_cache so each unique (src, dst) pair is only
# ever computed once across all episodes. 

_astar_cache = {}


def astar_distance(src_xy, dst_xy):
    """
    Shortest path distance between two normalised (x, y) coordinate pairs.

    Parameters
    ----------
    src_xy : (float, float)   normalised (x, y) from the observation
    dst_xy : (float, float)   normalised (x, y) from the observation

    Returns
    -------
    float  -- distance in normalised units (same scale as observation coords)
    """
    _check_bounds()

    src_rc = _to_pixel(*src_xy)
    dst_rc = _to_pixel(*dst_xy)

    key = (src_rc, dst_rc)
    if key in _astar_cache:
        return _astar_cache[key]

    src_rc = _nearest_road(*src_rc)
    dst_rc = _nearest_road(*dst_rc)

    if src_rc == dst_rc:
        _astar_cache[key] = 0.0
        return 0.0

    g_score = {src_rc: 0}

    def heuristic(r, c):
        return abs(r - dst_rc[0]) + abs(c - dst_rc[1])

    open_set = []
    heapq.heappush(open_set, (heuristic(*src_rc), src_rc))

    while open_set:
        _, current = heapq.heappop(open_set)

        if current == dst_rc:
            result = g_score[current] * _DELTA
            _astar_cache[key] = result
            return result

        for dr, dc in [(-1,0),(1,0),(0,-1),(0,1)]:
            nr, nc = current[0]+dr, current[1]+dc
            neighbour = (nr, nc)
            if 0 <= nr < MAP_HEIGHT and 0 <= nc < MAP_WIDTH and ROAD[nr, nc]:
                tentative_g = g_score[current] + 1
                if neighbour not in g_score or tentative_g < g_score[neighbour]:
                    g_score[neighbour] = tentative_g
                    f = tentative_g + heuristic(nr, nc)
                    heapq.heappush(open_set, (f, neighbour))

    penalty = np.sqrt(_MAX_X**2 + _MAX_Y**2)
    _astar_cache[key] = penalty
    return penalty


def clear_cache():
    """Free memory by clearing the A* cache."""
    _astar_cache.clear()


# ---------------------------------------------------------------------------
# 5. Feature extractor
# ---------------------------------------------------------------------------

def get_features(context):
    """
    Convert a raw DynamicPricingEnv context into a fixed-length numpy vector.
    ---------------------------------------------------
    context = (passenger_info, driver_info)
        passenger_info : np.array([x_o, y_o, x_f, y_f, alpha_passenger])
        driver_info    : tuple of np.arrays, each [x_d, y_d, alpha_driver]

    Feature vector  (8 elements)
    ----------------------------
    Index  Name                   Description
    -----  ---------------------  ------------------------------------------------
      0    trip_distance          A* distance: passenger origin -> destination.
                                  Longer trips justify higher prices.

      1    nearest_driver_dist    A* distance: nearest driver -> passenger origin.
                                  Far drivers are less likely to accept the ride.

      2    avg_driver_dist        Average A* distance of all drivers -> passenger.
                                  Reflects overall driver supply accessibility.

      3    alpha_passenger        Passenger price sensitivity (noisy estimate).
                                  Higher = willing to pay more per unit distance.

      4    avg_alpha_driver       Average driver price sensitivity.

      5    min_alpha_driver       Minimum driver alpha (most eager/cheapest driver).
                                  Most relevant for whether a booking succeeds.

      6    num_drivers            Number of nearby drivers, normalised by max (10).
                                  More supply -> lower price needed.

      7    price_gap              alpha_passenger - min_alpha_driver.
                                  Positive: passenger willing to pay more than the
                                  cheapest driver demands -> booking likely.
                                  Negative: expectations mismatched -> hard to book.
                                  Pre-computing this interaction helps linear models
                                  since they cannot multiply features on their own.

    All distance features normalised by map diagonal so they fall in [0, 1],
    same scale as the alpha features.

    Returns
    -------
    np.ndarray, shape (8,), dtype float32
    """
    _check_bounds()

    passenger_info, driver_info = context
    x_o, y_o = float(passenger_info[0]), float(passenger_info[1])
    x_f, y_f = float(passenger_info[2]), float(passenger_info[3])
    alpha_p   = float(passenger_info[4])

    diagonal = np.sqrt(_MAX_X**2 + _MAX_Y**2)

    trip_dist = astar_distance((x_o, y_o), (x_f, y_f)) / diagonal

    driver_distances, driver_alphas = [], []
    for driver in driver_info:
        x_d, y_d = float(driver[0]), float(driver[1])
        alpha_d  = float(driver[2])
        driver_distances.append(astar_distance((x_d, y_d), (x_o, y_o)) / diagonal)
        driver_alphas.append(alpha_d)

    nearest_driver_dist = float(min(driver_distances))
    avg_driver_dist     = float(np.mean(driver_distances))
    avg_alpha_driver    = float(np.mean(driver_alphas))
    min_alpha_driver    = float(min(driver_alphas))
    num_drivers_norm    = len(driver_info) / _MAX_DRIVERS
    price_gap           = alpha_p - min_alpha_driver

    return np.array([
        trip_dist,
        nearest_driver_dist,
        avg_driver_dist,
        alpha_p,
        avg_alpha_driver,
        min_alpha_driver,
        num_drivers_norm,
        price_gap,
    ], dtype=np.float32)


# ---------------------------------------------------------------------------
# 6. Sanity check  (run:  python features.py)
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import time

    print(f"Map loaded  : {MAP_WIDTH} x {MAP_HEIGHT} pixels")
    print(f"Road pixels : {ROAD.sum()} / {ROAD.size}  ({100*ROAD.mean():.1f}% road)\n")

    from RideSharing import DynamicPricingEnv
    env = DynamicPricingEnv()
    set_env_bounds(env)

    print(f"DELTA  : {_DELTA:.6f}")
    print(f"MAX_X  : {_MAX_X:.4f}")
    print(f"MAX_Y  : {_MAX_Y:.4f}\n")

    context, _ = env.reset()
    passenger_info, _ = context
    src = (float(passenger_info[0]), float(passenger_info[1]))
    dst = (float(passenger_info[2]), float(passenger_info[3]))

    t0 = time.time(); d1 = astar_distance(src, dst); t1 = time.time()
    d2 = astar_distance(src, dst);                   t2 = time.time()

    print(f"A* distance {src} -> {dst}: {d1:.5f} (normalised)")
    print(f"  First call  : {(t1-t0)*1000:.1f} ms")
    print(f"  Cached call : {(t2-t1)*1000:.2f} ms\n")

    features = get_features(context)
    names = [
        "trip_distance                (normalised)",
        "nearest_driver_distance      (normalised)",
        "avg_driver_distance          (normalised)",
        "passenger_alpha",
        "avg_driver_alpha",
        "min_driver_alpha",
        "num_drivers                  (normalised)",
        "price_gap (passenger_alpha - min_driver_alpha)",
    ]
    print("Feature vector (8 elements):")
    for name, val in zip(names, features):
        print(f"  {name:50s} = {val:.5f}")