import numpy as np

from typing import List, Tuple


def compute_cost_to_go(stepcosts: np.ndarray) -> np.ndarray:
    """ Compile a table of future totals from a table of instant costs.

    Parameters:
    -----------
    stepcosts : array of shape (N, M)
        A table of prices for standing on the various tiles of a grid.
    
    Returns:
    --------
    futurecosts : array of shape (N, M)
        A table of optimal-path costs. The value in cell (i, j) is the
        cost of reaching tile (N, M) by the cheapest path that only
        consists of steps that increase one or both coordinates by 1.
    """
    n, m = stepcosts.shape
    futurecosts = np.zeros([n, m], dtype=stepcosts.dtype)
    futurecosts[n - 1, m - 1] = 0.0
    for i in reversed(range(n - 1)):
        futurecosts[i, m - 1] = stepcosts[i, m - 1] + futurecosts[i + 1, m - 1]
    for j in reversed(range(m - 1)):
        futurecosts[n - 1, j] = stepcosts[n - 1, j] + futurecosts[n - 1, j + 1]

    for i in reversed(range(n - 1)):
        for j in reversed(range(m - 1)):
            mincost = min([
                        futurecosts[i + 1, j + 1],  # advance both
                        futurecosts[i + 1, j],  # advance i
                        futurecosts[i, j + 1],  # advance j
                        ])
            futurecosts[i, j] = stepcosts[i, j] + mincost

    return futurecosts


def compute_shortest_path(costs_to_go: np.ndarray) -> List[Tuple]:
    """ Compute an optimal path through a grid from the costs to go.
    
    Parameters:
    -----------
    futurecosts : array of shape (N, M)
        A table of smallest possible costs for reaching the goal tile
        (N - 1, M - 1) from any given tile (i, j).
    
    Returns:
    --------
    path : list of tuples
        A list of pairs (i, j) starting in (0, 0) and ending in (N, M).
        Each consecutive pair is either of the form (i, j) -> (i + 1, j),
        (i, j) -> (i, j + 1), or (i, j) -> (i + 1, j + 1). The computed
        path is the shortest path using only these moves.
    """
    n, m = costs_to_go.shape
    path = [(0, 0)]
    while path[-1] != (n - 1, m - 1):
        i, j = path[-1]
        if i == n - 1:
            path.append((i, j + 1))
        elif j == m - 1:
            path.append((i + 1, j))
        else:
            candidate_1 = costs_to_go[i + 1, j + 1]
            candidate_2 = costs_to_go[i + 1, j]
            candidate_3 = costs_to_go[i, j + 1]
            if candidate_1 <= candidate_2 and candidate_1 <= candidate_3:
                path.append((i + 1, j + 1))
            elif candidate_2 <= candidate_3:
                path.append((i + 1, j))
            else:
                path.append((i, j + 1))
    return path


def compute_straight_path(n, m, numsteps=None):
    """ Interpolate linearly between pairs (0, 0) and (n - 1, m - 1). """
    numsteps = max(n, m) if numsteps is None else numsteps
    isteps = np.linspace(0, n - 1, numsteps)
    jsteps = np.linspace(0, m - 1, numsteps)
    return np.transpose([isteps, jsteps]).round().astype(np.int32)


def compute_hinged_path(n, m):
    """ A list of pairs starting with (0, 0), (1, 1), (2, 2), ... """
    isteps = list(range(n))
    jsteps = list(range(m))
    if n <= m:
        isteps += [n - 1 for _ in range(m - n)]
    else:
        jsteps += [m - 1 for _ in range(n - m)]
    return np.transpose([isteps, jsteps])


def _test_that_cost_to_go_has_the_right_shape():
    n, m = np.random.randint(1, 10, size=2)
    dists = np.random.gamma(1, size=(n, m))
    forward = compute_cost_to_go(dists)
    assert forward.shape == dists.shape


def _test_that_cost_to_has_zero_in_the_corner():
    n, m = np.random.randint(1, 10, size=2)
    dists = np.random.gamma(1, size=(n, m))
    forward = compute_cost_to_go(dists)
    assert np.isclose(forward[-1, -1], 0)


def _test_that_path_has_the_right_start_and_end():
    n, m = np.random.randint(1, 10, size=2)
    dists = np.random.gamma(1, size=(n, m))
    forward = compute_cost_to_go(dists)
    path = compute_shortest_path(forward)
    assert path[0] == (0, 0)
    assert path[-1] == (n - 1, m - 1)


def _test_that_straight_path_is_not_cheaper_than_shortest_path():
    dists = np.random.gamma(1, size=(5, 5))
    straight_path_cost = dists.diagonal().sum()
    forward = compute_cost_to_go(dists)
    shortest_path_cost = forward[0, 0]
    assert shortest_path_cost <= straight_path_cost


def _test_that_shortest_path_is_cheaper_than_random_path():
    n, m = np.random.randint(1, 10, size=2)
    dists = np.random.gamma(1, size=(n, m))
    forward = compute_cost_to_go(dists)
    shortest_path_cost = forward[0, 0]
    i, j = 0, 0
    random_path_cost = 0.0
    while i < n - 1 and j < m - 1:
        random_path_cost += dists[i, j]
        if i == n - 1:
            j += 1
        elif j == m - 1:
            i += 1
        else:
            direction = np.random.choice(["N", "E", "NE"])
            if direction == "N":
                j += 1
            elif direction == "E":
                i += 1
            else:
                i += 1
                j += 1
    assert shortest_path_cost <= random_path_cost
