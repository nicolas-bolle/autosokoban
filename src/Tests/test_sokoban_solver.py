#!/usr/bin/env python3
"""Unit tests for sokoban_solver.py"""

# pylint: disable=protected-access

import unittest
import numpy as np

from src.Tests.boards import (
    node_trivial,
    node_simple1,
    node_simple2,
    node_simple3,
    node_open,
    node_uturn,
    node_tricky_uturn,
    node_1_1565730199_1,
    node_1_1565730199_2,
    node_1_1565730199_3,
    node_1_1565730199_4,
    node_1_1565730199_5,
    node_1_1565730199_11,
)

from src.sokoban_solver import (
    SokobanEdge,
    SokobanNode,
    Solver,
    BFSQueue,
    PriorityQueue,
)

from src.heuristics import (  # pylint: disable=unused-import
    f_priority_length,
    f_priority_l1_naive,
    f_priority_l1_symmetric_naive,
    f_priority_l1_matching,
    f_priority_dijkstra,
    f_priority_dijkstra_stuck_basic,
    f_priority_dlsb,
    get_f_priority_sbp,
)

from src.data_structures import Path


class TestSokobanEdge(unittest.TestCase):
    """Testing SokobanEdge"""

    def test_init_vector(self):
        """Testing an invalid vector is caught"""
        vector = (0, 0)
        point1 = (0, 0)
        point2 = (0, 0)
        point3 = (0, 0)

        with self.assertRaises(AssertionError):
            SokobanEdge(vector, point1, point2, point3)

    def test_init_points(self):
        """Testing invalid points are caught"""
        vector = (1, 0)
        point1 = (0, 0)
        point2 = (1, 0)
        point3 = (3, 0)

        with self.assertRaises(AssertionError):
            SokobanEdge(vector, point1, point2, point3)

        vector = (0, 1)
        point1 = (0, 0)
        point2 = (0, 2)
        point3 = (0, 3)

        with self.assertRaises(AssertionError):
            SokobanEdge(vector, point1, point2, point3)

    def test_hash_eq(self):
        """Testing hashing and equality"""

        edge1 = SokobanEdge((1, 0), (0, 0), (1, 0), (2, 0))
        edge2 = SokobanEdge((1, 0), (0, 0), (1, 0), (2, 0))
        edge3 = SokobanEdge((0, 1), (0, 0), (0, 1), (0, 2))

        self.assertIsInstance(hash(edge1), int)
        self.assertEqual(hash(edge1), hash(edge2))
        self.assertEqual(edge1, edge2)
        self.assertNotEqual(edge1, edge3)


class TestSokobanNode(unittest.TestCase):
    """Testing SokobanNode"""

    def test_init(self):
        """Testing invalid parameters are caught"""

        # invalid player
        airs = {(0, 0)}
        boxes = set()
        goals = set()
        player = (0, 1)

        with self.assertRaises(AssertionError):
            SokobanNode(airs, boxes, goals, player)

        # invalid boxes
        airs = {(0, 0)}
        boxes = {(0, 1)}
        goals = set()
        player = (0, 0)

        with self.assertRaises(AssertionError):
            SokobanNode(airs, boxes, goals, player)

        # invalid goals
        airs = {(0, 0)}
        boxes = set()
        goals = {(0, 1)}
        player = (0, 0)

        with self.assertRaises(AssertionError):
            SokobanNode(airs, boxes, goals, player)

        # boxes/goals mismatch
        airs = {(0, 0), (0, 1)}
        boxes = {(0, 1)}
        goals = set()
        player = (0, 0)

        with self.assertRaises(AssertionError):
            SokobanNode(airs, boxes, goals, player)

    def test_hash_eq(self):
        """Testing hashing and equality"""
        airs = {(0, 0), (0, 1), (0, 3)}
        boxes = set()
        goals = set()
        player = (0, 0)

        node1 = SokobanNode(airs, boxes, goals, player)
        node2 = SokobanNode(airs, boxes, goals, player)

        airs = {(0, 0), (0, 1), (0, 3)}
        boxes = set()
        goals = set()
        player = (0, 1)

        node3 = SokobanNode(airs, boxes, goals, player)

        airs = {(0, 0), (0, 1), (0, 3)}
        boxes = set()
        goals = set()
        player = (0, 3)

        node4 = SokobanNode(airs, boxes, goals, player)

        self.assertIsInstance(hash(node1), int)
        self.assertEqual(hash(node1), hash(node2))
        self.assertEqual(node1, node2)

        self.assertEqual(node1, node3)

        self.assertNotEqual(node1, node4)

    def test_str(self):
        """Testing str"""
        airs = {(0, 0), (1, 0), (2, 0), (3, 0)}
        boxes = {(1, 0), (3, 0)}
        goals = {(2, 0), (3, 0)}
        player = (0, 0)

        node1 = SokobanNode(airs, boxes, goals, player)

        str1 = (
            "█ █ █ █ █ █ █ █\n"
            "█ █ █ █ █ █ █ █\n"
            "█ █ ○ □ · ■ █ █\n"
            "█ █ █ █ █ █ █ █\n"
            "█ █ █ █ █ █ █ █"
        )

        self.assertEqual(str(node1), str1)

        airs = {(0, 0), (1, 0)}
        boxes = {(1, 0)}
        goals = {(0, 0)}
        player = (0, 0)

        node2 = SokobanNode(airs, boxes, goals, player)

        str2 = (
            "█ █ █ █ █ █\n"
            "█ █ █ █ █ █\n"
            "█ █ ● □ █ █\n"
            "█ █ █ █ █ █\n"
            "█ █ █ █ █ █"
        )

        self.assertEqual(str(node2), str2)

    def test_player_component(self):
        """Testing we discover the right player component"""
        airs = {(0, 0), (1, 0), (2, 0), (3, 0), (0, -2), (0, 1), (1, 1), (2, 1)}
        boxes = {(2, 0)}
        goals = {(1, 0)}
        player = (0, 0)

        node = SokobanNode(airs, boxes, goals, player)

        component = {(0, 0), (1, 0), (0, 1), (1, 1), (2, 1)}

        self.assertEqual(node.player_component, component)

    def test_solved(self):
        """Testing we can identify a solved puzzle"""
        airs = {(0, 0), (1, 0)}
        boxes = {(1, 0)}
        goals = {(1, 0)}
        player = (0, 0)

        node = SokobanNode(airs, boxes, goals, player)

        self.assertTrue(node.solved)

    def test_edges(self):
        """Testing we discover edges correctly"""
        airs = {(0, 0), (1, 0), (2, 0), (0, 1), (1, 1), (2, 1), (0, 2), (1, 2), (2, 2)}
        boxes = {(1, 1), (1, 2)}
        goals = {(1, 1), (1, 2)}
        player = (0, 0)

        node = SokobanNode(airs, boxes, goals, player)
        edges = node.edges

        edge1 = SokobanEdge((1, 0), (0, 1), (1, 1), (2, 1))
        edge2 = SokobanEdge((-1, 0), (2, 1), (1, 1), (0, 1))
        edge3 = SokobanEdge((1, 0), (0, 2), (1, 2), (2, 2))
        edge4 = SokobanEdge((-1, 0), (2, 2), (1, 2), (0, 2))

        self.assertEqual(len(edges), 4)
        self.assertEqual(edges[0], edge1)
        self.assertEqual(edges[1], edge2)
        self.assertEqual(edges[2], edge3)
        self.assertEqual(edges[3], edge4)

    def test_move(self):
        """Testing movement along an edge"""
        airs = {(0, 0), (1, 0), (2, 0)}
        boxes = {(1, 0)}
        goals = {(2, 0)}
        player = (0, 0)

        node_before_move = SokobanNode(airs, boxes, goals, player)

        edge = SokobanEdge((1, 0), (0, 0), (1, 0), (2, 0))

        airs = {(0, 0), (1, 0), (2, 0)}
        boxes = {(2, 0)}
        goals = {(2, 0)}
        player = (0, 0)

        node_after_move = SokobanNode(airs, boxes, goals, player)

        self.assertEqual(node_before_move.move(edge), node_after_move)
        self.assertEqual(node_before_move + edge, node_after_move)

    def test_unmove(self):
        """Testing unmovement along an edge"""
        airs = {(0, 0), (1, 0), (2, 0)}
        boxes = {(1, 0)}
        goals = {(2, 0)}
        player = (0, 0)

        node_before_move = SokobanNode(airs, boxes, goals, player)

        edge = SokobanEdge((1, 0), (0, 0), (1, 0), (2, 0))

        airs = {(0, 0), (1, 0), (2, 0)}
        boxes = {(2, 0)}
        goals = {(2, 0)}
        player = (0, 0)

        node_after_move = SokobanNode(airs, boxes, goals, player)

        self.assertEqual(node_after_move.unmove(edge), node_before_move)
        self.assertEqual(node_after_move - edge, node_before_move)

    def test_check_move(self):
        """Testing move checking"""
        airs = {(0, 0), (1, 0), (2, 0)}
        boxes = {(1, 0)}
        goals = {(2, 0)}
        player = (0, 0)

        node_before_move = SokobanNode(airs, boxes, goals, player)

        edge = SokobanEdge((1, 0), (1, 0), (2, 0), (3, 0))

        self.assertFalse(node_before_move._check_move(edge))

    def test_check_unmove(self):
        """Testing unmove checking"""
        edge = SokobanEdge((1, 0), (1, 0), (2, 0), (3, 0))

        airs = {(0, 0), (1, 0), (2, 0)}
        boxes = {(2, 0)}
        goals = {(2, 0)}
        player = (0, 0)

        node_after_move = SokobanNode(airs, boxes, goals, player)

        self.assertFalse(node_after_move._check_unmove(edge))


class TestPuzzles(unittest.TestCase):
    """Testing that we can solve various test puzzles"""

    def test_node_trivial(self):
        """Testing node_trivial"""
        node = node_trivial
        solver = Solver(BFSQueue(), node)
        solver.solve()
        self.assertEqual(solver.status_code, 200)
        self.assertEqual(solver.n_iters_overall, 0)
        self.assertEqual(len(solver.solution), 0)

    def test_node_simple1(self):
        """Testing node_simple1"""
        node = node_simple1
        solver = Solver(BFSQueue(), node)
        solver.solve()
        self.assertEqual(solver.status_code, 200)
        self.assertEqual(solver.n_iters_overall, 1)
        self.assertEqual(len(solver.solution), 1)

    def test_node_simple2(self):
        """Testing node_simple2"""
        node = node_simple2
        solver = Solver(PriorityQueue(f_priority_dlsb), node)
        solver.solve()
        self.assertEqual(solver.status_code, 200)
        self.assertEqual(solver.n_iters_overall, 3)
        self.assertEqual(len(solver.solution), 3)

    def test_node_simple3(self):
        """Testing node_simple3"""
        node = node_simple3
        solver = Solver(PriorityQueue(f_priority_dlsb), node)
        solver.solve()
        self.assertEqual(solver.status_code, 200)
        self.assertEqual(solver.n_iters_overall, 2)
        self.assertEqual(len(solver.solution), 2)

    def test_node_open(self):
        """Testing node_open"""
        node = node_open
        solver = Solver(PriorityQueue(f_priority_dlsb), node)
        solver.solve()
        self.assertEqual(solver.status_code, 200)
        # self.assertEqual(solver.n_iters_overall, 0)
        # self.assertEqual(len(solver.solution), 0)

    def test_node_uturn(self):
        """Testing node_uturn"""
        node = node_uturn
        solver = Solver(PriorityQueue(f_priority_dlsb), node)
        solver.solve()
        self.assertEqual(solver.status_code, 200)
        # self.assertEqual(solver.n_iters_overall, 0)
        # self.assertEqual(len(solver.solution), 0)

    def test_node_tricky_uturn(self):
        """Testing node_tricky_uturn"""
        node = node_tricky_uturn
        solver = Solver(PriorityQueue(f_priority_dlsb), node)
        solver.solve()
        self.assertEqual(solver.status_code, 200)
        # self.assertEqual(solver.n_iters_overall, 0)
        # self.assertEqual(len(solver.solution), 0)

    def test_node_1_1565730199_1(self):
        """Testing node_1_1565730199_1"""
        node = node_1_1565730199_1
        solver = Solver(PriorityQueue(f_priority_dlsb), node)
        solver.solve()
        self.assertEqual(solver.status_code, 200)
        # self.assertEqual(solver.n_iters_overall, 0)
        # self.assertEqual(len(solver.solution), 0)

    def test_node_1_1565730199_2(self):
        """Testing node_1_1565730199_2"""
        node = node_1_1565730199_2
        solver = Solver(PriorityQueue(f_priority_dlsb), node)
        solver.solve()
        self.assertEqual(solver.status_code, 200)
        # self.assertEqual(solver.n_iters_overall, 0)
        # self.assertEqual(len(solver.solution), 0)

    def test_node_1_1565730199_3(self):
        """Testing node_1_1565730199_3"""
        node = node_1_1565730199_3
        solver = Solver(PriorityQueue(f_priority_dlsb), node)
        solver.solve()
        self.assertEqual(solver.status_code, 200)
        # self.assertEqual(solver.n_iters_overall, 0)
        # self.assertEqual(len(solver.solution), 0)

    def test_node_1_1565730199_4(self):
        """Testing node_1_1565730199_4"""
        node = node_1_1565730199_4
        solver = Solver(PriorityQueue(f_priority_dlsb), node)
        solver.solve()
        self.assertEqual(solver.status_code, 200)
        # self.assertEqual(solver.n_iters_overall, 0)
        # self.assertEqual(len(solver.solution), 0)

    def test_node_1_1565730199_5(self):
        """Testing node_1_1565730199_5"""
        node = node_1_1565730199_5
        solver = Solver(PriorityQueue(f_priority_dlsb), node)
        solver.solve()
        self.assertEqual(solver.status_code, 200)
        # self.assertEqual(solver.n_iters_overall, 0)
        # self.assertEqual(len(solver.solution), 0)

    def test_node_1_1565730199_11(self):
        """Testing node_1_1565730199_11"""
        node = node_1_1565730199_11
        solver = Solver(PriorityQueue(f_priority_dlsb), node)
        solver.solve(max_seconds=1)
        self.assertEqual(solver.status_code, 301)
        # FIXME still haven't solved this one!
        # self.assertEqual(solver.n_iters_overall, 0)
        # self.assertEqual(len(solver.solution), 0)

    def test_f_priority_sbp(self):
        """Testing SBP heuristic"""
        node = node_1_1565730199_5
        f_priority_sbp = get_f_priority_sbp(node)
        solver = Solver(PriorityQueue(f_priority_sbp), node)
        solver.solve()
        self.assertEqual(solver.status_code, 200)

        # priority of the original node
        path = Path(node, [], node)
        self.assertEqual(f_priority_sbp(path), 12)

        # distance of an impossible box/goal pair
        self.assertEqual(f_priority_sbp.dist((0, 0), (5, 0), (0, 1)), np.inf)


if __name__ == "__main__":
    unittest.main()
