import numpy as np

class Board:
    merge_array = None  # Class variable to store the merge array

    def __init__(self, board: int = None, num_moves: int = 0):
        if board is None:
            self.board = np.array([0], dtype=np.uint64)
            # self._spawn_initial_tiles()
        else:
            self.board = np.array([board], dtype=np.uint64)
        if Board.merge_array is None:
            Board._initialize_merge_array()

        self.total_moves = num_moves

    def __str__(self):
        return str(self.get_2048_board())

    @classmethod
    def _initialize_merge_array(cls):
        # Precompute the merge array for all 16-bit values
        arr = np.arange(0, 0xffff + 1, 1, dtype=np.uint16)
        cls._compute_merge(arr)
        cls.merge_array = arr

    @staticmethod
    def _compute_merge(arr):
        # Swipes the board to the left

        # Extract each 4-bit nibble
        n0 = (arr >> 0) & 0xF
        n1 = (arr >> 4) & 0xF
        n2 = (arr >> 8) & 0xF
        n3 = (arr >> 12) & 0xF

        # Stack nibbles into a 2D array for vectorized operations
        tiles = np.stack([n3, n2, n1, n0], axis=1)

        # The way 2048 works is that you shift all the tiles to the left
        # and then merge the tiles if they are equal. Then you shift the
        # tiles to the left again. This means if there are 4 tiles in a row
        # that are equal, the first two tiles will merge, then the last two
        # tiles will merge. And then tiles are shifted to the left again,
        # but the merged tiles are not merged again.
        # So 4 4 4 4 -> 8 8 0 0 and not 16 0 0 0
        # And 8 4 4 0 -> 8 8 0 0 and not 16 0 0 0

        # Swipe left: Shift non-zero tiles to the left.
        # Gives rows of [1, 4, 0, 2] -> [0, 0, 1, 0]
        non_zero = tiles == 0
        tiles_sorted = np.zeros_like(tiles)
        # Sort the non-zero tiles to the left by index
        # So that [1, 4, 0, 2] -> [0, 0, 1, 0]
        # is sorted by index to [0, 1, 3, 2]
        indices = np.argsort(non_zero, axis=1)
        # Sort the tiles by the indices to move the non-zero tiles to the left
        # and the zero tiles to the right
        tiles_sorted = tiles[np.arange(tiles.shape[0])[:, None], indices[:,:4]]
        tiles = tiles_sorted

        # Merge tiles by incrementing duplicates
        
        for i in range(3):
            # Get mask of rows where the current tile is equal to the next tile
            merge_mask = (tiles[:, i] == tiles[:, i + 1]) & (tiles[:, i] != 0)
            # Increment the current tile if the next tile is equal
            tiles[merge_mask, i] += 1
            # Set the next tile to 0
            tiles[merge_mask, i + 1] = 0

        # Shift non-zero tiles to the left again
        non_zero = tiles == 0
        tiles_sorted = np.zeros_like(tiles)
        indices = np.argsort(non_zero, axis=1)
        tiles_sorted = tiles[np.arange(tiles.shape[0])[:, None], indices[:,:4]]
        tiles = tiles_sorted


        # Reassemble the 16-bit rows
        merged = ((tiles[:,0] << 12) |
                  (tiles[:,1] << 8) |
                  (tiles[:,2] << 4) |
                  (tiles[:,3] << 0))

        # Update the array in place
        arr[:] = merged
        
    def merge(self, rows):
        # Merge the rows
        return Board.merge_array[rows]

    def swipe_left(self):
        # Convert the board to a 16-bit array with each row as a 16-bit value
        # View creates an array where the first value (16 bit number) is the last
        # row of the board.
        board = self.board.view(np.uint16)
        # Merge the rows by using the merge array
        merged = self.merge(board)
        board[:] = merged

    def swipe_right(self):
        # Convert the board to a 16-bit array with each row as a 16-bit value
        # View creates an array where the first value (16 bit number) is the last
        # row of the board.
        board = self.board.view(np.uint16)
        # Reverse the rows before swiping left
        board[:] = ((board >> 12) | 
                    ((board >> 4) & 0x00F0) | 
                    ((board << 4) & 0x0F00) | 
                    (board << 12))
        # Swipe the board to the left
        self.swipe_left()
        # Reverse the rows on the board using bitwise operations
        board[:] = ((board >> 12) |
                    (board << 12) |
                    ((board & 0x0F00) >> 4) |
                    ((board & 0x00F0) << 4))
                
    def swipe_up(self):
        # Convert the board to a 16-bit array with each row as a 16-bit value
        # View creates an array where the first value (16 bit number) is the last
        # row of the board.
        # LAST ROW IS FIRST VALUE
        board = self.board.view(np.uint16)
        # Modify the board so each 16-bit value is a column instead of a row
        # using bitwise operations
        new_board = np.zeros_like(board)
        # Extract each column from each 16-bit value
        # First column is the first nibble of each 16-bit value
        n0 = (board >> 0) & 0xF
        # Second column is the second nibble of each 16-bit value
        n1 = (board >> 4) & 0xF
        n2 = (board >> 8) & 0xF
        n3 = (board >> 12) & 0xF

        # n0 n1 n2 n3 are the columns of the board
        new_board[0] = (n0[3] << 12) | (n0[2] << 8) | (n0[1] << 4) | n0[0]
        new_board[1] = (n1[3] << 12) | (n1[2] << 8) | (n1[1] << 4) | n1[0]
        new_board[2] = (n2[3] << 12) | (n2[2] << 8) | (n2[1] << 4) | n2[0]
        new_board[3] = (n3[3] << 12) | (n3[2] << 8) | (n3[1] << 4) | n3[0]

        # Merge the columns by using the merge array
        merged = self.merge(new_board)

        # Convert the merged columns back to rows
        # Extract each column from each 16-bit value
        n0 = (merged >> 0) & 0xF
        n1 = (merged >> 4) & 0xF
        n2 = (merged >> 8) & 0xF
        n3 = (merged >> 12) & 0xF

        # n0 n1 n2 n3 are the columns of the board
        board[0] = (n0[3] << 12) | (n0[2] << 8) | (n0[1] << 4) | n0[0]
        board[1] = (n1[3] << 12) | (n1[2] << 8) | (n1[1] << 4) | n1[0]
        board[2] = (n2[3] << 12) | (n2[2] << 8) | (n2[1] << 4) | n2[0]
        board[3] = (n3[3] << 12) | (n3[2] << 8) | (n3[1] << 4) | n3[0]

    def swipe_down(self):
        # Convert the board to a 16-bit array with each row as a 16-bit value
        # View creates an array where the first value (16 bit number) is the last
        # row of the board.
        # LAST ROW IS FIRST VALUE
        board = self.board.view(np.uint16)
        # Flip the board upside down
        board[:] = board[::-1]
        # Swipe the board to the left up
        self.swipe_up()
        # Flip the board upside down again
        board[:] = board[::-1]

    def swipe(self, direction):
        # Move the board in a direction
        if direction == "left":
            self.swipe_left()
        elif direction == "right":
            self.swipe_right()
        elif direction == "up":
            self.swipe_up()
        elif direction == "down":
            self.swipe_down()

    def move(self, direction):
        # Move the board in a direction
        self.swipe(direction)
        self.spawn_random_tile()
        self.total_moves += 1
    def move2(self, direction):
        # Move the board in a direction
        self.swipe(direction)
        self.total_moves += 1

    def get_2048_board(self):
        # Get the 2048 board from the 64-bit board
        new_board = np.zeros((4, 4), dtype=np.uint64)
        board_value = int(self.board[0])  # Convert to Python integer for bit operations
        # Row
        for i in range(4):
            # Column
            for j in range(4):
                val = ((board_value >> (i * 16 + j * 4)) & 0xF)
                if val > 0:
                    new_board[3 - i, 3 - j] = 2 ** val
        return new_board
    
    def _spawn_initial_tiles(self):
        # Spawn two initial tiles
        # Because the board is initialized to 0, we can just pick two random tiles
        # and insert a 2 or 4 into the tile randomly.
        
        # Pick two random tiles from the 16 tiles
        spawn_indices = np.random.choice(16, 2, replace=False)

        # 90% chance of 2 (value 1), 10% chance of 4 (value 2)
        new_values = np.random.choice([1, 2], 2, p=[0.9, 0.1]).astype(np.uint64)

        # Create shift amounts for each spawn position
        shifts = spawn_indices * 4

        # Ensure shifts are of type np.uint64
        shifts = shifts.astype(np.uint64)

        # Create the values to OR with the board
        spawn_values = np.left_shift(new_values.astype(np.uint64), shifts)

        # Update the board using NumPy's bitwise OR
        self.board = np.bitwise_or(self.board, spawn_values[0])
        self.board = np.bitwise_or(self.board, spawn_values[1])
           
    def spawn_random_tile(self):
        # Insert a random tile into the board
        # Each tile is 4 bits, so we need to find the empty tiles
        # and insert a 2, 4, 8, 16, or 32 into the tile randomly.
        # The probabilities are: 80% for 2, 12% for 4, 4% for 8, 2% for 16, and 2% for 32.

        # Convert board to 4x4 array of 4-bit values
        board_value = int(self.board[0])
        tiles = np.zeros(16, dtype=np.uint8)
        for i in range(16):
            tiles[i] = (board_value >> (i * 4)) & 0xF
        
        # Find empty tiles
        empty_indices = np.where(tiles == 0)[0]
        
        if len(empty_indices) > 0:
            # Choose random empty tile
            spawn_index = np.uint64(np.random.choice(empty_indices))
            # Determine the new value based on the given probabilities
            new_value = np.uint64(np.random.choice([1, 2, 3, 4, 5], p=[0.8, 0.12, 0.04, 0.02, 0.02]))
            
            # Update the board with the new tile
            self.board[0] |= np.uint64(new_value << np.uint64(spawn_index * 4))


    def set_value(self, cell, value):
        # Set the value of a specific cell
        # Cell is a tuple of the row and column
        # Value is the value of the tile
        # Get value as 2 -> 1, 4 -> 2 etc. for the 2048 board
        if value == 0:
            # Clear the cell
            self.board[0] &= np.uint64(~(np.uint64(0xF) << np.uint64((3 - cell[0]) * 16 + (3 - cell[1]) * 4)))
        else:
            value = np.uint64(np.log2(value))
            # Clear the cell first
            self.board[0] &= np.uint64(~(np.uint64(0xF) << np.uint64((3 - cell[0]) * 16 + (3 - cell[1]) * 4)))
            # Set the value of the cell
            self.board[0] |= np.uint64(value << np.uint64((3 - cell[0]) * 16 + (3 - cell[1]) * 4))

    def can_swipe_left(self):
        test_board = self.board.copy()
        self.swipe_left()
        changed = not np.array_equal(test_board, self.board)
        self.board = test_board
        return changed

    def can_swipe_right(self):
        test_board = self.board.copy()
        self.swipe_right()
        changed = not np.array_equal(test_board, self.board)
        self.board = test_board
        return changed

    def can_swipe_up(self):
        test_board = self.board.copy()
        self.swipe_up()
        changed = not np.array_equal(test_board, self.board)
        self.board = test_board
        return changed

    def can_swipe_down(self):
        test_board = self.board.copy()
        self.swipe_down()
        changed = not np.array_equal(test_board, self.board)
        self.board = test_board
        return changed
    
    def get_valid_moves(self):
        # Get the valid moves
        moves = []
        if self.can_swipe_left():
            moves.append("left")
        if self.can_swipe_right():
            moves.append("right")
        if self.can_swipe_up():
            moves.append("up")
        if self.can_swipe_down():
            moves.append("down")
        return moves

    def is_game_over(self):
        # Check if any swipe is possible
        if self.can_swipe_left() or self.can_swipe_right() or self.can_swipe_up() or self.can_swipe_down():
            return False
        return True
    
    def copy(self):
        return Board(int(self.board[0]), self.total_moves)
    
    def score(self):
        # Get the score of the board
        # This is done by repeatedly adding the value of the tiles
        # and dividing by 2 until the value is 0
        # This is because score is increased by the new value of the tile
        # when two tiles are merged
        score = 0
        game_board = self.get_2048_board()
        while np.any(game_board):
            score += np.sum(game_board)
            game_board = game_board // 2
            # Get rid of any 2s in the board as they are not part of the score
            game_board[game_board == 2] = 0
        return score
    
    def get_open_cells(self):
        # Get the indices of the open cells
        game_board = self.get_2048_board()
        return np.argwhere(game_board == 0)
    
    def place_tile(self, cell, value):
        # Place a tile in the board
        # Cell is a tuple of the row and column
        # Value is the value of the tile
        # Get value as 2 -> 1, 4 -> 2 etc. for the 2048 board
        value = np.uint64(np.log2(value))
        # Set the value of the tile using bitwise operations
        # Clear the cell first
        self.board[0] &= np.uint64(~(np.uint64(0xF) << np.uint64((3 - cell[0]) * 16 + (3 - cell[1]) * 4)))
        # Set the value of the cell
        self.board[0] |= np.uint64(value << np.uint64((3 - cell[0]) * 16 + (3 - cell[1]) * 4))


if __name__ == "__main__":
    board = Board()
    # board.board = np.array([0x1000_0100_0010_0001], dtype=np.uint64)
    print(board.board)
    board.swipe_down()
    board.swipe_right()
    board.swipe_up()
    board.swipe_left()
    print(board.board)
    print(hex(board.board[0]))
    print(board.get_2048_board())
    board.place_tile((3, 1), 16)
    print(board.get_2048_board())

