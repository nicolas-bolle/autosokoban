"""The Board class for a sokoban board state"""

from src.types import Self, Point, Points, Vector, Push, Pushes
import numpy as np

PRINTING_SYMBOLS = {
    'wall': '█',
    'air': ' ',
    'box': '□',
    'goal': '·',
    'box and goal': '■',
    'player': 'x',
}

WALL_THICKNESS = 2

class Board:
    """A sokoban board: air spaces, boxes, goals, the player
    "air" really means "non wall", so these include locations of blocks, goals, and the player 
    """

    airs: Points
    boxes: Points
    goals: Points
    player: Point

    _player_component: Points
    _possible_pushes: Pushes

    _hash: int
    _str: str

    def __init__(self, airs: Points, boxes: Points, goals: Points, player: Point):
        assert player in airs
        assert len(boxes - airs) == 0
        assert len(goals - airs) == 0
        assert len(boxes) == len(goals)
        
        self.airs = airs
        self.boxes = boxes
        self.goals = goals
        self.player = player

        self._player_component = None
        self._possible_pushes = None

        self._hash = None
        self._str = None

    def __repr__(self):
        return f'Board({self.airs}, {self.boxes}, {self.goals}, {self.player})'

    def __str__(self):
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

                    c = PRINTING_SYMBOLS['wall']
                    if is_air:
                        c = PRINTING_SYMBOLS['air']
                    if is_box:
                        c = PRINTING_SYMBOLS['box']
                    if is_goal:
                        c = PRINTING_SYMBOLS['goal']
                    if is_box and is_goal:
                        c = PRINTING_SYMBOLS['box and goal']
                    if is_player:
                        c = PRINTING_SYMBOLS['player']
                        
                    arr[j][i] = c

            # form the string, going over rows in reverse order so the y orientation looks right
            self._str = '\n'.join([' '.join(row) for row in reversed(arr)])
            
        return self._str

    def __eq__(self, board: Self) -> bool:
        """Check equality against another board, up to non-push player moves"""
        return (self.airs == board.airs) and (self.boxes == board.boxes) and (self.goals == board.goals) and (board.player in self.player_component)

    def __hash__(self):
        """Hash of the set, to achieve the same sense of equality as __eq__"""
        if not self._hash:
            self._hash = hash((hash(frozenset(self.airs)), hash(frozenset(self.boxes)), hash(frozenset(self.goals)), hash(frozenset(self.player_component))))
        return self._hash

    @property
    def player_component(self) -> Points:
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
                    if not loc in self.airs:
                        continue
                    if loc in self.boxes:
                        continue

                    # otherwise, it's a new location we can visit!
                    self._player_component.add(loc)
                    queue.append(loc)
            
        return self._player_component

    @property
    def possible_pushes(self) -> Pushes:
        """List out the possible pushes a player can do
        Could implement iterating over self.player_component instead of boxes if that one is shorter
        """
        if not self._possible_pushes:
            self._possible_pushes = []
            
            # iterate over boxes we can push
            for x_box, y_box in self.boxes:
                # and possible player pushes to do to this box
                movement_vectors = ((1, 0), (-1, 0), (0, 1), (0, -1))
                for dx, dy in movement_vectors:
                    # record a valid push
                    push: Push = ((dx, dy), (x_box - dx, y_box - dy), (x_box, y_box), (x_box + dx, y_box + dy))
                    if self._check_valid_push(push):
                        self._possible_pushes.append(push)
        
        return self._possible_pushes

    def _check_valid_push(self, push: Push) -> bool:
        """Check if a particular Push is valid on our Board"""
        (dx, dy), (x1, y1), (x2, y2), (x3, y3) = push
        assert x1 + dx == x2
        assert x2 + dx == x3
        assert y1 + dy == y2
        assert y2 + dy == y3
        return ((x1, y1) in self.player_component) and ((x2, y2) in self.boxes) and ((x3, y3) in self.airs) and ((x3, y3) not in self.boxes)

    def push(self, push: Push) -> Self:
        """Return the Board resulting from a particular Push"""
        assert self._check_valid_push(push)
        
        # start with current board, only making copies of the boxes (which are the only thing that changes)
        airs = self.airs
        boxes = self.boxes.copy()
        goals = self.goals

        # modify to reflect the push
        _, point1, point2, point3 = push
        boxes.remove(point2)
        boxes.add(point3)
        player = point2
        
        return Board(airs, boxes, goals, player)

    @property
    def solved(self) -> bool:
        """Whether the sokoban is solved"""
        return self.boxes == self.goals