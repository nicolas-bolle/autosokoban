"""A solver for sokoban puzzles"""

from src.types import Self, Point, Points, Vector, Push, Pushes
from src.board import Board

def print_solve_sequence(board, pushes):
    print(board)
    for push in pushes:
        board = board.push(push)
        print()
        print(board)
    
class BoardHistory:
    """Bundles a Board with a history of the pushes that got us where we are"""

    board: Board
    pushes: Pushes
    
    def __init__(self, board: Board, pushes: Pushes = None):
        if pushes is None:
            pushes = []
        self.board = board
        self.pushes = pushes

    def __repr__(self):
        return f'BoardHistory({repr(self.board)}, {repr(self.pushes)})'

    @property
    def possible_pushes(self) -> Pushes:
        """Exposing the board property"""
        return self.board.possible_pushes

    @property
    def solved(self) -> bool:
        """Exposing the board property"""
        return self.board.solved

    def push(self, push: Push) -> Self:
        """Applying a push"""
        board = self.board.push(push)
        pushes = self.pushes + [push]
        return BoardHistory(board, pushes)

class Solver:
    """Solver for sokoban puzzles (specified as Board objects)
    Uses BoardHistory objects to track the pushes made, to output the solution
    """
    board: Board
    
    def __init__(self, board: Board):
        self.board = board

    def __repr__(self):
        return f'Solver({repr(self.board)})'

    def solve(self) -> BoardHistory:
        """Solve the board"""
        boardhistory = BoardHistory(self.board)
        
        if boardhistory.solved:
            return boardhistory

        # queue of boards to explore and a record of where we've been
        queue = [boardhistory]
        boards_visited = {boardhistory.board}
        
        while queue:
            boardhistory_old = queue.pop()
            for push in boardhistory_old.possible_pushes:
                boardhistory = boardhistory_old.push(push)
                
                if boardhistory.board in boards_visited:
                    continue
                
                if boardhistory.solved:
                    return boardhistory

                queue.append(boardhistory)
                boards_visited.add(boardhistory.board)
            
        return 'Unable to solve!'