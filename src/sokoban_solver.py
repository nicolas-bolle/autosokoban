"""Classes for a sokoban solver"""

from typing import Self
import numpy as np

from src.data_structures import Edge, Node, Path, Solver, BFSQueue, PriorityQueue  # pylint: disable=unused-import


PRINTING_SYMBOLS = {
    "wall": "█",
    "air": " ",
    "box": "□",
    "player": "○",
    "goal": "·",
    "goal and box": "■",
    "goal and player": "●",
}

WALL_THICKNESS = 2


type Point = tuple[int, int]
type Vector = tuple[int, int]


class SokobanEdge(Edge):
    """A box-pushing move in a sokoban puzzle"""

    vector: Vector  # player movement
    point1: Point  # original player location
    point2: Point  # destination player location / original box location
    point3: Point  # destination box location

    def __init__(self, vector: Vector, point1: Point, point2: Point, point3: Point):
        self.vector = vector
        self.point1 = point1
        self.point2 = point2
        self.point3 = point3
        assert self.valid

    def __repr__(self):
        return f"SokobanEdge({repr(self.vector)}, {repr(self.point1)}, {repr(self.point2)}, {repr(self.point3)})"

    def __str__(self):
        return f"Move {self.vector} from {self.point1} to {self.point2}"

    def __hash__(self):
        return hash((self.vector, self.point1, self.point2, self.point3))

    @property
    def valid(self):
        """Checking the edge we represent is valid"""
        dx, dy = self.vector
        x1, y1 = self.point1
        x2, y2 = self.point2
        x3, y3 = self.point3

        return (
            ((dx, dy) in ((1, 0), (-1, 0), (0, 1), (0, -1)))
            and (x1 + dx == x2)
            and (x2 + dx == x3)
            and (y1 + dy == y2)
            and (y2 + dy == y3)
        )


class SokobanNode(Node):  # pylint: disable= too-many-instance-attributes
    """A sokoban board: air spaces, boxes, goals, the player
    "air" really means "non wall", so these include locations of blocks, goals, and the player
    Two boards are considered equal if one can reach the other via non-box-pushing moves
    """

    airs: set[Point]
    boxes: set[Point]
    goals: set[Point]
    player: set[Point]

    _player_component: set[Point]
    _edges: list[SokobanEdge]

    _hash: int
    _str: str

    def __init__(
        self, airs: set[Point], boxes: set[Point], goals: set[Point], player: Point
    ):
        assert player in airs
        assert len(boxes - airs) == 0
        assert len(goals - airs) == 0
        assert len(boxes) == len(goals)

        self.airs = airs
        self.boxes = boxes
        self.goals = goals
        self.player = player

        self._player_component = None
        self._edges = None

        self._hash = None
        self._str = None

    def __repr__(self):
        return f"SokobanNode({self.airs}, {self.boxes}, {self.goals}, {self.player})"

    def __str__(self):  # pylint: disable=too-many-locals
        if not self._str:
            # figure out coordinate bounds
            x_min, x_max = np.inf, -np.inf
            y_min, y_max = np.inf, -np.inf
            for x, y in self.airs:
                x_min = min(x_min, x)
                x_max = max(x_max, x)
                y_min = min(y_min, y)
                y_max = max(y_max, y)

            # compute ranges and offsets
            x_range = x_max - x_min + 1 + 2 * WALL_THICKNESS
            x_offset = x_min - WALL_THICKNESS
            y_range = y_max - y_min + 1 + 2 * WALL_THICKNESS
            y_offset = y_min - WALL_THICKNESS

            # make an array to store everything
            # indexed as [j][i] since the rows will be x coords
            arr = [[None for _ in range(x_range)] for _ in range(y_range)]

            # populate the array appropriately
            for i in range(x_range):
                x = i + x_offset
                for j in range(y_range):
                    y = j + y_offset
                    point = (x, y)

                    is_air = point in self.airs
                    is_box = point in self.boxes
                    is_goal = point in self.goals
                    is_player = point == self.player

                    c = PRINTING_SYMBOLS["wall"]
                    if is_air:
                        c = PRINTING_SYMBOLS["air"]
                    if is_box:
                        c = PRINTING_SYMBOLS["box"]
                    if is_player:
                        c = PRINTING_SYMBOLS["player"]
                    if is_goal:
                        c = PRINTING_SYMBOLS["goal"]
                    if is_goal and is_box:
                        c = PRINTING_SYMBOLS["goal and box"]
                    if is_goal and is_player:
                        c = PRINTING_SYMBOLS["goal and player"]

                    arr[j][i] = c

            # form the string, going over rows in reversed order so the y orientation looks right
            self._str = "\n".join([" ".join(row) for row in reversed(arr)])

        return self._str

    def __hash__(self):
        """Hash for determining equality"""
        if not self._hash:
            self._hash = hash(
                (
                    hash(frozenset(self.airs)),
                    hash(frozenset(self.boxes)),
                    hash(frozenset(self.goals)),
                    hash(frozenset(self.player_component)),
                )
            )
        return self._hash

    @property
    def player_component(self) -> set[Point]:
        """The locations the player can reach"""
        if not self._player_component:
            # explore the locations in the player's connected component
            self._player_component = set()
            queue = []

            loc = self.player
            self._player_component.add(loc)
            queue.append(loc)

            while queue:
                x, y = queue.pop()
                candidate_locs = ((x + 1, y), (x - 1, y), (x, y + 1), (x, y - 1))
                for loc in candidate_locs:
                    # skip if we've already visited it
                    if loc in self._player_component:
                        continue

                    # skip if it's not reachable by the player
                    if loc not in self.airs:
                        continue
                    if loc in self.boxes:
                        continue

                    # otherwise, it's a new location we can visit!
                    self._player_component.add(loc)
                    queue.append(loc)

        return self._player_component

    @property
    def solved(self) -> bool:
        """Whether the node is in a "solved" state"""
        return self.boxes == self.goals

    @property
    def edges(self) -> list[SokobanEdge]:
        """Edges we can move along from the node"""
        if not self._edges:
            self._edges = []

            # iterate over boxes we can push
            for x_box, y_box in self.boxes:
                # and possible player pushes to do to this box
                for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    # record a valid edge
                    edge = SokobanEdge(
                        (dx, dy),
                        (x_box - dx, y_box - dy),
                        (x_box, y_box),
                        (x_box + dx, y_box + dy),
                    )
                    if self._check_move(edge):
                        self._edges.append(edge)

        return self._edges

    def move(self, edge: SokobanEdge) -> Self:
        """Movement along an edge, returning the new node"""
        assert self._check_move(edge)

        # start with current node, only making copies of the boxes (which are the only thing that changes)
        airs = self.airs
        boxes = self.boxes.copy()
        goals = self.goals

        # modify to reflect the move
        boxes.remove(edge.point2)
        boxes.add(edge.point3)
        player = edge.point2

        return SokobanNode(airs, boxes, goals, player)

    def unmove(self, edge: SokobanEdge) -> Self:
        """Undo movement along an edge, returning the new node"""
        assert self._check_unmove(edge)

        # start with current node, only making copies of the boxes (which are the only thing that changes)
        airs = self.airs
        boxes = self.boxes.copy()
        goals = self.goals

        # modify to reflect the unmove
        boxes.remove(edge.point3)
        boxes.add(edge.point2)
        player = edge.point1

        return SokobanNode(airs, boxes, goals, player)

    def _check_move(self, edge: SokobanEdge) -> bool:
        """Check validity of moving along an edge"""
        return (
            (edge.point1 in self.player_component)
            and (edge.point2 in self.boxes)
            and (edge.point3 in self.airs)
            and (edge.point3 not in self.boxes)
        )

    def _check_unmove(self, edge: SokobanEdge) -> bool:
        """Check validity of unmoving along an edge"""
        return (
            (edge.point1 in self.airs)
            and (edge.point1 not in self.boxes)
            and (edge.point2 in self.player_component)
            and (edge.point3 in self.boxes)
        )
