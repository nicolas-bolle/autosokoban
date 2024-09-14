"""Data structures for graph searches"""

from abc import ABC, abstractmethod
from collections.abc import Callable
from typing import Self, Any
import time
import heapq
import numpy as np


class Edge(ABC):
    """Abstract graph edge
    Lightweight, with usage defined in Node
    """

    @abstractmethod
    def __repr__(self):
        pass

    @abstractmethod
    def __str__(self):
        pass

    @abstractmethod
    def __hash__(self):
        pass

    def __eq__(self, edge):
        return hash(self) == hash(edge)


class Node(ABC):
    """Abstract graph node"""

    @abstractmethod
    def __repr__(self):
        pass

    @abstractmethod
    def __str__(self):
        pass

    @abstractmethod
    def __hash__(self):
        """Hash for determining equality
        Allows avoiding node revists in the solver
        """

    def __eq__(self, node):
        return hash(self) == hash(node)

    def __add__(self, edge: Edge):
        return self.move(edge)

    def __sub__(self, edge: Edge):
        return self.unmove(edge)

    @property
    @abstractmethod
    def solved(self) -> bool:
        """Whether the node is in a "solved" state"""

    @property
    @abstractmethod
    def edges(self) -> list[Edge]:
        """Edges we can move along from the node"""

    @abstractmethod
    def move(self, edge: Edge) -> Self:
        """Movement along an edge, returning the new node"""

    @abstractmethod
    def unmove(self, edge: Edge) -> Self:
        """Undo movement along an edge, returning the new node"""

    @abstractmethod
    def _check_move(self, edge: Edge) -> bool:
        """Check validity of moving along an edge"""

    @abstractmethod
    def _check_unmove(self, edge: Edge) -> bool:
        """Check validity of unmoving along an edge"""


class Path:
    """Lightweight object recording a path in the graph of nodes"""

    start_node: Node
    edges: list[Edge]
    end_node: Node

    def __init__(self, start_node, edges, end_node):
        self.start_node = start_node
        self.edges = edges
        self.end_node = end_node

    def __repr__(self):
        return (
            f"Path({repr(self.start_node)}, {repr(self.edges)}, {repr(self.end_node)})"
        )

    def __str__(self):
        out_list = []
        if len(self.edges):
            out_list.append(str(self.start_node))
            out_list.append("")
            out_list.append("to")
            out_list.append("")
            out_list.append(str(self.end_node))
            out_list.append("")
            out_list.append("via")
            out_list.append("")
            for edge in self.edges:
                out_list.append(str(edge))
        else:
            out_list.append(str(self.start_node))
            out_list.append("")
            out_list.append("with no moves")
        return "\n".join(out_list)

    def get_str_end_node(self):
        """String for just the end node"""
        return str(self.end_node)

    def __len__(self):
        return len(self.edges)


class Queue(ABC):
    """Abstract queue for paths"""

    @abstractmethod
    def __repr__(self):
        pass

    def __str__(self):
        return "\n\n----------\n\n".join([str(path) for path in self])

    def get_str_end_nodes(self):
        """String for just the end nodes"""
        return "\n\n----------\n\n".join([path.get_str_end_node() for path in self])

    @abstractmethod
    def __len__(self):
        """Length of the queue"""

    @abstractmethod
    def __getitem__(self, i):
        """Getting the ith element of the queue, without modification"""

    @abstractmethod
    def __iter__(self):
        """Queue iterable, without altering the queue"""

    @property
    def empty(self) -> bool:
        """True if the queue is empty"""
        return len(self) == 0

    @abstractmethod
    def push(self, path: Path):
        """Push a path onto the queue"""

    @abstractmethod
    def pop(self) -> Path:
        """Pop the top path off the queue"""


class Solver:  # pylint: disable=too-many-instance-attributes
    """Graph search solver: initialize with queue object and starting node"""

    # starting node of the problem instance
    start_node: Node

    # queue of path objects
    queue: Queue

    # hashes of nodes covered
    # updated when adding to the queue
    nodes_visited_hashes: set[int]

    # solver progress tracking on the the overall optimization
    # times recorded as floats following time.time()
    solving_started: bool = False
    solving_finished: bool = False
    solved: bool = False
    solution: Path = None
    t_start_overall: float = None
    t_end_overall: float = None
    n_seconds_overall: int = 0
    n_iters_overall: int = 0
    n_revisited_overall: int = 0

    # solver progress tracking on the current "sprint" of solving
    t_start_sprint: float = None
    t_end_sprint: float = None
    n_seconds_sprint: int = None
    n_iters_sprint: int = 0
    n_revisited_sprint: int = 0
    status_code = 300

    def __init__(self, queue: Queue, start_node: Node):
        # initialize fields
        self.start_node = start_node
        self.queue = queue
        self.nodes_visited_hashes = set()

        # define the starting point for the search by "sniffing" the null path
        path = Path(self.start_node, [], self.start_node)
        self.sniff(path)

    def __repr__(self):
        return f"{type(self).__name__}({repr(self.queue)}, {repr(self.start_node)})"

    def __str__(self):
        # FIXME add more info?
        return f"{type(self.queue).__name__} solver"

    def sniff(self, path: Path):
        """ "Sniff" a path to see if it's worth exploring"""

        # check if we already have a solution
        if self.solved:
            return

        # check if this path is solved
        if path.end_node.solved:
            self.solved = True
            self.solution = path
            return

        # check if the end node of this path has been explored
        _hash = hash(path.end_node)
        if _hash in self.nodes_visited_hashes:
            self.n_revisited_sprint += 1
            self.n_revisited_overall += 1
            return

        # record a visit to the end node of this path and add it to the queue
        self.nodes_visited_hashes.add(_hash)
        self.queue.push(path)

    def explore(self, path: Path):
        """ "Explore" a path: sniff out the results of adding on the next edges"""
        # FIXME consider implementing path ordering tracking so priority queues can preserve it

        if self.solved:
            return
        for edge in path.end_node.edges:
            start_node = path.start_node
            edges = path.edges + [edge]
            end_node = path.end_node + edge
            path_new = Path(start_node, edges, end_node)
            self.sniff(path_new)
            if self.solved:
                return

    def step(self):
        """One step in the graph search process: explore the top of the queue"""
        assert not self.queue.empty, "Queue is depleted"
        path = self.queue.pop()
        self.explore(path)

    def steps(self, steps: int):
        """Do multiple steps"""
        for _ in range(steps):
            self.step()

    def solve(self, max_seconds: int | None = 10, max_iters: int | None = None) -> int:
        """Run steps of the solving process and return status codes

        Status codes: divided into classes
            200-299: solution found
                200: solution found
            300-399: iteration terminated with queue remaining
                A solution may exist, but wasn't found with the current iteration constraints
                300: no iterations have been done yet
                301: maximum time reached
                302: maximum iterations reached
            400-499: queue depleted without finding a solution
                400: queue depleted without finding a solution

        Parameters
        ----------
            max_seconds: int, optional
                Maximum number of seconds to iterate (default is 10)
                If unspecified, runs without bound on iteration time
            max_iters: int, optional
                Maximum number of iterations to run (default is None)
                If unspecified, runs without bound on number of iterations

        Returns
        -------
            status_code: int
                Status code for the result of the iteration, see above for details
        """
        if max_seconds is None:
            max_seconds = np.inf
        if max_iters is None:
            max_iters = np.inf

        # set up progress tracking fields
        t_start = time.time()

        self.t_start_sprint = t_start
        self.t_end_sprint = None
        self.n_seconds_sprint = 0
        self.n_iters_sprint = 0
        self.n_revisited_sprint = 0
        self.status_code = None

        if not self.solving_started:
            self.solving_started = True
            self.solving_finished = False
            self.t_start_overall = t_start
            self.t_end_overall = None
            self.n_seconds_overall = 0
            self.n_iters_overall = 0
            self.n_revisited_overall = 0

        # separate fields for tracking iteration termination
        t_max = t_start + max_seconds
        n_iters = 0

        # loop until iteration is terminated
        while True:
            if self.solved:
                status_code = 200
                break

            if self.queue.empty:
                status_code = 400
                break

            if time.time() > t_max:
                status_code = 301
                break

            if n_iters > max_iters:
                status_code = 302
                break

            self.step()
            n_iters += 1

        # update progress tracking fields
        t_end = time.time()

        self.t_end_sprint = t_end
        self.n_seconds_sprint = self.t_end_sprint - self.t_start_sprint
        self.n_iters_sprint = n_iters
        self.status_code = status_code

        if (200 <= self.status_code < 300) or (400 <= self.status_code < 500):
            self.solving_finished = True
        self.t_end_overall = t_end
        self.n_seconds_overall += self.n_seconds_sprint
        self.n_iters_overall += self.n_iters_sprint

        return self.status_code


class BFSQueue(Queue):
    """Breadth-first-search queue"""

    queue: list[Path]

    def __init__(self):
        self.queue = []

    def __repr__(self):
        return "BFSQueue()"

    def __len__(self):
        """Length of the queue"""
        return len(self.queue)

    def __getitem__(self, i):
        """Getting the ith element of the queue, without modification"""
        return self.queue[i]

    def __iter__(self):
        """Queue iterable, without altering the queue"""
        return iter(self.queue)

    def push(self, path: Path):
        """Push a path onto the queue"""
        self.queue.append(path)

    def pop(self) -> Path:
        """Pop the top path off the queue"""
        return self.queue.pop(0)


class PriorityQueue(Queue):
    """Priority queue: initialize with priority function accepting a path
    Uses heapq to maintain self.heap as a (min) heap

    Elements are tuples (priority, order, path)
        priority (Any): priority of the element, with lower values being higher priority
            Commonly int, tuple of int, or float
            Any priority type is allowed as long as it admits comparisons
        order (int): increments for each element added, serving as a tiebreak to avoid comparisons on path objects
        path (Path): the actual item being stored
    """

    # list of (priority, order, path), see docstring for explanations
    heap: list[tuple[Any, int, Path]]

    order: int

    f_priority: Callable[[Path], Any]

    def __init__(self, f_priority: Callable[[Path], Any]):
        self.heap = []
        self.order = 0
        self.f_priority = f_priority

    def __repr__(self):
        return f"{type(self).__name__}()"

    def __len__(self):
        """Length of the queue"""
        return len(self.heap)

    def __getitem__(self, i):
        """Getting the ith element of the queue, without modification"""
        return self.heap[i][-1]

    def __iter__(self):
        """Queue iterable, without altering the queue"""
        for item in self.heap:
            yield item[-1]

    def push(self, path: Path):
        """Push a path onto the queue
        An item with priority of just None or np.inf is discarded
        """
        priority = self.f_priority(path)
        if (priority is None) or (priority == np.inf):
            return
        item = (priority, self.order, path)
        heapq.heappush(self.heap, item)
        self.order += 1

    def pop(self) -> Path:
        """Pop the top path off the queue"""
        return heapq.heappop(self.heap)[-1]
