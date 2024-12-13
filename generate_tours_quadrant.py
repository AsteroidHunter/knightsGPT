"""
This script generates a million unique solutions 
from the first quadrant of an 8 x 8 chessboard.

@author: akash
"""

import os
import random
import polars as pl
from tqdm import tqdm
from datetime import date

random.seed(347923473)

current_date_formatted = date.today().strftime("%Y%m%d")

# dictionary mapping 2D to 1D indices
pos_to_int = {i: j for i, j in zip(
    [(i, j) for i in range(8) for j in range(8)],
    range(64)
)}

# defining possible offsets
possible_steps_offset = [
    (2, 1), (1, 2), (-1, 2), (-2, 1),
    (-2, -1), (-1, -2), (1, -2), (2, -1)
]

def valid_move(
    x, 
    y, 
    board
):
    return 0 <= x < 8 and 0 <= y < 8 and board[y][x] == -1

def degree_of_move(
    x, 
    y, 
    board
):
    count = 0
    for dx, dy in possible_steps_offset:
        if valid_move(x + dx, y + dy, board):
            count += 1
    return count

def make_moves(
    x, 
    y, 
    board
):
    moves = []
    for dx, dy in possible_steps_offset:
        new_x, new_y = x + dx, y + dy
        if valid_move(new_x, new_y, board):
            moves.append((new_x, new_y))
    
    # applying a two-degree heuristic
    moves.sort(key=lambda move: degree_of_move(move[0], move[1], board))

    # randomizing the top two moves
    random.shuffle(moves[:2])
    
    return moves

def kt(
    initial_position=(0, 0), 
    solutions_limit=1000
):
    board = [
        [-1 for _ in range(8)] 
        for _ in range(8)
    ]

    # defining the starting position
    x, y = initial_position
    
    move_number = 0
    board[y][x] = move_number

    # adding the initial position to the path
    path = [(x, y)]
    
    solutions = []

    def backtrack(x, y, move_number):
        # when 64 moves are played, a tuple of the solution is saved
        if move_number == 8 * 8:
            solutions.append(tuple(path))
            pbar.update(1)

            # terminate the recursion when the given number of solutions are found
            if len(solutions) >= solutions_limit:
                return True
            return False

        for new_x, new_y in make_moves(x, y, board):
            board[new_y][new_x] = move_number + 1
            path.append((new_x, new_y))

            # perform backtracking if the knight hits a dead end
            if backtrack(new_x, new_y, move_number + 1):
                return True
            board[new_y][new_x] = -1
            
            path.pop()
        return False

    # setting a progress counter
    with tqdm(total=solutions_limit, desc="Solutions found") as pbar:
        backtrack(x, y, move_number)

    # return solutions as a tuple of tuples
    return tuple(tuple(pos_to_int[i] for i in solutions[j]) for j in range(len(solutions)))


def save_as_parquet(
    tours,
    output_directory="./data/"
):
    """

    :param tours: The completed Knight's tours as a list of sequences
    :param output_directory: The directory to save the solutions
    :return: A compressed parquet file with the solved tours
    """
    total_moves = len(tours[0])

    # int8 schema to reduce file size
    as_df = pl.DataFrame(
        tours,
        schema={f"move_{i}": pl.Int8 for i in range(total_moves)},
        orient="row"
    )

    # file path and name with metadata
    file_path = (
        f"{output_directory}"
        f"/tours_"
        f"{8}x{8}_"
        f"{1000000}_"
        f"{current_date_formatted}_"
        f"recursive_"
        f"{''.join(str(i) for i in current_date_formatted)}.parquet"
    )

    try:
        # if the file already exists, concatenate new results and save it
        if os.path.exists(file_path):
            existing_df = pl.read_parquet(file_path)
            as_df = pl.concat([existing_df, as_df])

        as_df.write_parquet(
            file_path,
            compression="zstd"
        )

    except Exception as e:
        print(f"File could not be saved due to the following error: {e}")


# list of positions from which tours should be generated
pos_to_sample = [
    (0,0), (0, 1), (0, 2), (0, 3), 
    (1, 0), (1, 1), (1, 2), (1, 3), 
    (2, 0), (2, 1), (2, 2), (2, 3), 
    (3, 0), (3, 1), (3, 2), (3, 3)
]


tours_to_generate = 1_000_000

for i in pos_to_sample:
    save_as_parquet(kt(i, solutions_limit=tours_to_generate))