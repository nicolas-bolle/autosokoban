"""Basic data types for sokoban puzzles"""

from typing import Self

type Point = tuple[int, int]
type Points = set[Point]
type Vector = tuple[int, int]

# a Push is the movement vector, old player location, new player location / old box location, new box location
type Push = tuple[Vector, Point, Point, Point]
type Pushes = list[Push]