"""Heuristic functions for PriorityQueue"""

import numpy as np
from scipy.optimize import linprog, linear_sum_assignment

from src.data_structures import Solver, BFSQueue
from src.sokoban_solver import Point, Vector, SokobanEdge, SokobanNode, Path  # pylint: disable=unused-import


def optimal_transport_cost(cost_matrix: np.ndarray) -> float:
    """Cost of the optimal transport of n sources to n targets,
    with cost i to j given by cost_matrix[i,j]
    For scipy.optimize.linprog, the variable x corresponds to cost_matrix.flatten()
    """
    assert len(cost_matrix.shape) == 2
    assert cost_matrix.shape[0] == cost_matrix.shape[1]
    n = len(cost_matrix)

    # trivial case
    if n == 1:
        return cost_matrix[0, 0]

    # specify LP
    c = cost_matrix.flatten()

    A_eq = []  # pylint: disable=invalid-name
    for i in range(n):
        temp = np.zeros((n, n))
        temp[i, :] = 1
        row1 = temp.flatten()

        temp = np.zeros((n, n))
        temp[:, i] = 1
        row2 = temp.flatten()

        A_eq.extend([row1, row2])

    b_eq = np.ones(2 * n)

    # solve LP
    res = linprog(c, A_eq=A_eq, b_eq=b_eq, bounds=(0, 1))

    return res.fun


def grid_shortest_distances(loc: Point, locs: set[Point]) -> dict[Point, int]:
    """Calculate the shortest distance between an intial loc and all available locs
    Movements are along 4 cardinal directions on an integer lattice

    Special case of Dijkstra's algorithm, which makes the implementation easier
    i.e. The first time we visit a node gives the shortest distance to it
    This means we just need to "spread" out through the lattice
    """
    distances = {_loc: np.inf for _loc in locs}

    # the vanguard will be points that were just reached for the first time
    distances[loc] = 0
    vanguard = [loc]
    distance = 0

    # loop until there is no vanguard (nowhere left to spread)
    while vanguard:
        # spread the vanguard
        _vanguard = [
            _loc
            for i, j in vanguard
            for _loc in ((i + 1, j), (i - 1, j), (i, j + 1), (i, j - 1))
        ]
        distance += 1

        # update distances and whittle down the vanguard
        # to locs reached for the first time
        vanguard = []
        for _loc in _vanguard:
            if distances.get(_loc, -1) == np.inf:
                distances[_loc] = distance
                vanguard.append(_loc)

    return distances


def check_for_stuck_boxes_basic(node: SokobanNode) -> bool:
    """Basic check for if any boxes are "strongly stuck", return True if so and False otherwise
    A box is strongly stuck if it is both
        Not on a goal
        Strongly unpushable: both the horizontal and veritical pushes are blocked by walls

    FIXME there's fancier versions of this:
    1. Consider "conditionally unpushable" boxes which are blocked by other boxes
    2. Consider boxes that could be pushed if the player gets "behind" them, but that space is blocked off
        (Potentially blocked off by another box)
    """
    # check if each box is strongly stuck
    for box in node.boxes:
        # if it's on a goal, it is not stuck
        if box in node.goals:
            continue

        # check which directions are blocked
        _box = (box[0] + 1, box[1])
        blocked_r = _box not in node.airs
        _box = (box[0] - 1, box[1])
        blocked_l = _box not in node.airs
        _box = (box[0], box[1] + 1)
        blocked_u = _box not in node.airs
        _box = (box[0], box[1] - 1)
        blocked_d = _box not in node.airs

        # translate into which push directions are blocked
        blocked_horizontal = blocked_r or blocked_l
        blocked_vertical = blocked_u or blocked_d

        # see if we can't do any pushes
        if blocked_horizontal and blocked_vertical:
            return True

    # reach here if none of the boxes were stuck
    return False


def f_priority_length(path: Path) -> int:
    """Priority by the length of a path
    A priority queue using this priority function is a BFS queue
    """
    return len(path)


def f_priority_l1_naive(path: Path) -> tuple[int, int]:
    """Priority by the sum of naive l1 distances of boxes to goals
    Distances are naive since they may map multiple boxes to the same goal
    """
    priority = 0
    for box in path.end_node.boxes:
        dist = np.inf
        for goal in path.end_node.goals:
            _dist = abs(box[0] - goal[0]) + abs(box[1] - goal[1])
            dist = min(dist, _dist)
        priority += dist
    return priority, f_priority_length(path)


def f_priority_l1_symmetric_naive(path: Path) -> tuple[int, int]:
    """Priority by the sum of naive l1 distances of boxes to goals and goals to boxes"""
    priority = 0

    for box in path.end_node.boxes:
        dist = np.inf
        for goal in path.end_node.goals:
            _dist = abs(box[0] - goal[0]) + abs(box[1] - goal[1])
            dist = min(dist, _dist)
        priority += dist

    for goal in path.end_node.goals:
        dist = np.inf
        for box in path.end_node.boxes:
            _dist = abs(box[0] - goal[0]) + abs(box[1] - goal[1])
            dist = min(dist, _dist)
        priority += dist

    return priority, f_priority_length(path)


def f_priority_l1_matching(path: Path) -> tuple[float, int]:
    """Priority by the sum of l1 distances in the optimal box/goal matching"""
    cost_matrix = [
        [abs(box[0] - goal[0]) + abs(box[1] - goal[1]) for box in path.end_node.boxes]
        for goal in path.end_node.goals
    ]

    cost = optimal_transport_cost(np.array(cost_matrix))

    return cost, f_priority_length(path)


def f_priority_dijkstra(path: Path) -> tuple[float, int]:
    """Priority by the sum of shortest path distances in the optimal box/goal matching"""
    cost_matrix = []
    for box in path.end_node.boxes:
        distances = grid_shortest_distances(box, path.end_node.airs)
        cost_matrix.append([distances[goal] for goal in path.end_node.goals])

    cost = optimal_transport_cost(np.array(cost_matrix))

    return cost, f_priority_length(path)


def f_priority_dijkstra_stuck_basic(path: Path) -> tuple[float, int]:
    """Dijkstra priority but with the "stuck" pruning condition"""
    if check_for_stuck_boxes_basic(path.end_node):
        return np.inf
    return f_priority_dijkstra(path)


def f_priority_dlsb(path: Path) -> int:
    """Weighted Dijkstra + length with basic "stuck" pruning"""
    if check_for_stuck_boxes_basic(path.end_node):
        return np.inf
    d, _l = f_priority_dijkstra(path)
    return 4 * d + _l


class SBPPriority:
    """Single box push priority
    Callable object instead of a function, to enable caching
    Infinite if it's impossible to match up boxes to goals
    Otherwise, it's the minimum number of moves necessary to do it where
        each box is handled independently (on an otherwise blank board)
    """

    # starting node
    node0: SokobanNode

    # cache of distances, for each box + goal + player setup
    dist_cache: dict[int, float]

    def __init__(self, node: SokobanNode):
        self.node0 = node
        self.dist_cache = {}

    def __call__(self, path: Path) -> float:
        """Compute the priority for a given path"""
        assert self.node0 == path.start_node
        node = path.end_node

        # cost matrix
        cost_matrix = np.array(
            [
                [self.dist(box, goal, node.player) for goal in node.goals]
                for box in node.boxes
            ]
        )

        try:
            # compute the minimum cost matching
            idx, jdx = linear_sum_assignment(cost_matrix)

            # return the cost
            return cost_matrix[idx, jdx].sum()

        except ValueError as e:
            # infeasible problem: couldn't match up boxes and goals
            assert str(e) == "cost matrix is infeasible"
            return np.inf

    def dist(self, box: Point, goal: Point, player: Point):
        """Distance between the given box and goal, given the player location"""
        # the node representing this distance calculation
        node = SokobanNode(self.node0.airs, {box}, {goal}, player)

        _hash = hash(node)
        if _hash not in self.dist_cache:
            # compute if not already computed
            solver = Solver(BFSQueue(), node)
            solver.solve()

            # cases depending on the status code
            match solver.status_code:
                case 200:
                    # success, so record appropriate distance for each node on the winning path
                    for d, node in enumerate(reversed(solver.solution.nodes())):
                        self.dist_cache[hash(node)] = d
                case 400:
                    # fail, so record infinite distance for each node visited
                    for _sub_hash in solver.nodes_visited_hashes:
                        self.dist_cache[_sub_hash] = np.inf
                case _:
                    assert False, "Unexpected status code"

        return self.dist_cache[_hash]


def get_f_priority_sbp(node: SokobanNode):
    """Function for building the object"""
    return SBPPriority(node)
