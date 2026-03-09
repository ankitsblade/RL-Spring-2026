import numpy as np
from PIL import Image
import heapq


#White pixels (>=128) = road (walkable)
#Black pixels (<128)  = building (blocked)]
_map_image  = Image.open("map_agent.png").convert("L")
_map_array  = np.array(_map_image)
MAP_HEIGHT, MAP_WIDTH = _map_array.shape
ROAD = _map_array >= 128

_DELTA       = None
_MAX_DRIVERS = None
_MAX_X       = None
_MAX_Y       = None


def set_env_bounds(env):

    global _DELTA, _MAX_X, _MAX_Y, _MAX_DRIVERS
    passenger_high = env.observation_space[0].high
    _MAX_X = float(passenger_high[0])
    _MAX_Y = float(passenger_high[1])
    _DELTA       = _MAX_X / MAP_WIDTH
    _MAX_DRIVERS = float(env.MaxDrivers)


def _check_bounds():
    if _DELTA is None:
        raise RuntimeError("Call set_env_bounds(env) once before using features.")


def _to_pixel(x, y):
    col = int(round(float(x) / _DELTA))
    row = int(round(float(y) / _DELTA))
    col = max(0, min(col, MAP_WIDTH  - 1))
    row = max(0, min(row, MAP_HEIGHT - 1))
    return row, col


def _nearest_road(row, col):

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


_astar_cache = {}


def astar_distance(src_xy, dst_xy):
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
    _astar_cache.clear()



def get_features(context):
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