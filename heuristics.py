import numpy as np
import binary_puzzle as bp

def score_heuristic(board: bp.Board) -> int:
    # Sum all of the values in the board
    return board.score()

def open_cells_heuristic(board: bp.Board) -> int:
    # This heuristic will return the number of open cells
    # This is equivalent to maximizing the number of merges
    # that can be done as if there is more open cells, there
    # must have been more merges
    return np.sum(board.get_2048_board() == 0)

def max_tile_heuristic(board: bp.Board) -> int:
    # This heuristic will return the maximum tile value
    return np.log2(np.max(board.get_2048_board()))

def tile_sum_heuristic(board: bp.Board) -> int:
    # This heuristic will return the weighted sum of all the tile values
    return np.sum(board.get_2048_board() ** 1.01)

"""
The following heuristics will return the heuristic value if the game is not 
over and 0 if the game is over
"""
def score_and_gamover_heuristic(board: bp.Board) -> int:
    # This heuristic will return the score if the game is not over
    # and 0 if the game is over
    if board.is_game_over():
        return -100000
    return score_heuristic(board)

def open_cells_and_gamover_heuristic(board: bp.Board) -> int:
    # This heuristic will return the number of open cells if the game is not over
    # and 0 if the game is over
    if board.is_game_over():
        return -100000
    return open_cells_heuristic(board)

def max_tile_and_gamover_heuristic(board: bp.Board) -> int:
    # This heuristic will return the maximum tile value if the game is not over
    # and 0 if the game is over
    if board.is_game_over():
        return -100000
    return max_tile_heuristic(board)

def tile_sum_and_gamover_heuristic(board: bp.Board) -> int:
    # This heuristic will return the weighted sum of all the tile values if the game is not over
    # and a large negative number if the game is over
    if board.is_game_over():
        return -100000
    return tile_sum_heuristic(board)


