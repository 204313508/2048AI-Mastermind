import binary_puzzle as bp
import numpy as np
from visual import GameVisual
import time
import math
import random
import heuristics

class Node:
    def __init__(self, board: bp.Board, parent=None, move=None):
        self.board = board.copy()
        self.parent = parent
        self.move = move  # Move that led to this node
        self.children = []
        self.wins = 0
        self.visits = 0
        self.untried_moves = board.get_valid_moves()

    def ucb1(self, exploration=0.1, normalizing_factor=1):
        if self.visits == 0:
            return float('inf')
        return ((self.wins / normalizing_factor) / self.visits) + exploration * math.sqrt(math.log(self.parent.visits) / self.visits)

    def add_child(self, move):
        new_board = self.board.copy()
        new_board.move(move)
        child = Node(new_board, parent=self, move=move)
        self.untried_moves.remove(move)
        self.children.append(child)
        return child

class MCTSBoard:
    def __init__(self, board: bp.Board, simulation_time=1.0, heuristic=None, exploration=0.1, greedy_heuristic=None):
        self.board = board
        self.simulation_time = simulation_time
        if heuristic is None:
            self.heuristic = heuristics.score_heuristic
        else:
            self.heuristic = heuristic
        self.exploration = exploration
        self.greedy_heuristic = greedy_heuristic
        self.normalizing_factor = 1

    def get_best_move(self) -> str:
        root = Node(self.board)
        end_time = time.time() + self.simulation_time

        while time.time() < end_time:
            # Selection
            node = root
            while node.untried_moves == [] and node.children:
                node = max(node.children, key=lambda n: n.ucb1(self.exploration, normalizing_factor=self.normalizing_factor))

            # Expansion
            if node.untried_moves:
                move = random.choice(node.untried_moves)
                node = node.add_child(move)

            # Simulation
            sim_board = node.board.copy()
            moves = sim_board.get_valid_moves()
            
            if self.greedy_heuristic:
                # Use greedy heuristic for simulation
                while moves:
                    best_move = None
                    best_h = float('-inf')
                    for move in moves:
                        test_board = sim_board.copy()
                        test_board.move(move)
                        h = self.greedy_heuristic(test_board)
                        if h > best_h:
                            best_h = h
                            best_move = move
                    sim_board.move(best_move)
                    moves = sim_board.get_valid_moves()
            else:
                # Use random moves for simulation
                while moves:
                    sim_board.move(random.choice(moves))
                    moves = sim_board.get_valid_moves()

            # Backpropagation
            score = self.heuristic(sim_board)
            if score > self.normalizing_factor:
                self.normalizing_factor = score
            while node:
                node.visits += 1
                node.wins += score
                node = node.parent

        # Print the number of visits for each child
        # print("Number of visits for each child:")
        # for child in root.children:
        #     print(child.move, child.visits)
        # Choose best move based on most visits
        return max(root.children, key=lambda c: c.visits).move if root.children else None

    def take_best_move(self) -> bool:
        move = self.get_best_move()
        if move is None:
            return False
        # print(f"Taking move {move}")
        self.board.move(move)
        return True

    def __str__(self):
        return str(self.board)

class VisualMCTS(GameVisual):
    def __init__(self, mcts_board: MCTSBoard, delay=1000):
        super().__init__()
        self.board = mcts_board.board
        self.mcts_board = mcts_board
        self.delay = delay
        self.update_grid_cells()
        self.after(self.delay, self.ai_move)
        self.mainloop()

    def ai_move(self):
        if self.mcts_board.take_best_move():
            self.update_grid_cells()
            if self.board.is_game_over():
                self.show_game_over()
            else:
                self.after(self.delay, self.ai_move)

if __name__ == '__main__':
    # Test with tile sum game over heuristic
    board = bp.Board()
    mcts_board = MCTSBoard(board, simulation_time=0.1, heuristic=heuristics.tile_sum_heuristic, exploration=0.1)
    visual = VisualMCTS(mcts_board, delay=1)

    

