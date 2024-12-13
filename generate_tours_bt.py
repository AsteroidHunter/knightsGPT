"""
@author: akash
"""

import os
import random
import argparse
import numpy as np
import polars as pl
from tqdm import tqdm
import multiprocessing
from datetime import date

random.seed(347923473)
np.random.seed(347923473)

current_date_formatted = date.today().strftime("%Y%m%d")

# setting arguments that could be passed when running the script
parser = argparse.ArgumentParser()
parser.add_argument(
    '--generate',
    type=int,
    help="Number of Knight's Tour solutions to generate",
    required=True
)

parser.add_argument(
    '--board_length',
    type=int,
    help="Length of the chessboard",
    default=8,
    required=False
)

parser.add_argument(
    '--board_height',
    type=int,
    help="Height of the chessboard",
    default=8,
    required=False
)

parser.add_argument(
    '--savedir',
    type=str,
    help="The name of the folder where the generated tours will be saved",
    default="data",
    required=False
)

parser.add_argument(
    '--existing_tours',
    type=str,
    help="Name of parquet file with existing Knight's Tour solutions; supplement this to avoid generating already generated solutions.",
    required=False
)

args = parser.parse_args()

if not os.path.exists(args.savedir):
    os.makedirs(args.savedir)

# dictionary outlining the eight possible Knight's moves
possible_steps_offset = {
    "up_and_right": [1, -2],
    "up_and_left": [-1, -2],

    "right_and_up": [2, -1],
    "right_and_down": [2, 1],

    "left_and_up": [-2, -1],
    "left_and_down": [-2, 1],

    "down_and_right": [1, 2],
    "down_and_left": [-1, 2],
}

# dictionary to convert (x, y) coordinates to board indices
pos_to_int = {i: j for i, j in zip(
    [(i, j) for i in range(args.board_length) for j in range(args.board_height)],
    range(args.board_length * args.board_height)
)}


def valid_move(
        x,
        y,
        board
):
    """

    :param x: The row number of the knight's position
    :param y: The column number of the knight's position
    :param board: The chessboard as a 2 x 2 list of lists
    :return: True if the move is valid, False otherwise
    """
    return (
            0 <= x < args.board_length and
            0 <= y < args.board_height and
            board[y][x] == -1
    )


def degree_of_the_move(
        x_index,
        y_index,
        board
):
    """

    :param x_index: The row number of the knight's position
    :param y_index: The column number of the knight's position
    :param board: The chessboard as a 2 x 2 list of lists
    :return: The total number of subsequent moves possible
    """
    count = 0

    for x, y in possible_steps_offset.values():
        new_x, new_y = x_index + x, y_index + y

        if valid_move(new_x, new_y, board):
            count += 1

    return count


def get_moves(
    x, 
    y, 
    board
):
    """

    :param x: The row number of the knight's position
    :param y: The column number of the knight's position
    :param board: The chessboard as a 2 x 2 list of lists
    :return: A list of possible moves from the (x, y) position
    """
    possible_moves = []
    
    for x_prime, y_prime in possible_steps_offset.values():
        new_x, new_y = x + x_prime, y + y_prime
        
        if valid_move(new_x, new_y, board):
            possible_moves.append((new_x, new_y))
    
    return possible_moves


def prioritize_moves(
    possible_moves, 
    board
):
    """

    :param possible_moves: The row number of the knight's position
    :param board: The chessboard as a 2 x 2 list of lists
    :return: Possible moves randomized and arranged according to Warnsdorff's heuristic
    """
    random.shuffle(possible_moves)
    possible_moves.sort(
        key=lambda move: degree_of_the_move(move[0], move[1], board)
    )
    return possible_moves


def solve_knights_tour_bt(
    initial_coordinate
):
    """

    :param initial_coordinate: A tuple containing the x, y coordinates of the starting position
    :return: If successful, a tuple containing a solved Knight's tour
    """
    
    # create board
    board = [[-1 for _ in range(args.board_length)] for _ in range(args.board_height)]
    
    # setting initial position and move number
    x, y = initial_coordinate
    move_number = 0
    
    # mark initial position as visited
    board[y][x] = move_number
    path = [(x, y)]

    next_moves = get_moves(x, y, board)
    next_moves = prioritize_moves(next_moves, board)

    # creating a log that helps track which moves from which positions have been attempted
    move_logs = [{
        "position": (x, y),
        "possible_moves": next_moves,
        "index": 0
    }]

    while len(move_logs) != 0:
        # if the Knight traverses the entire board, return the solution as indices
        if len(path) == args.board_length * args.board_height:
            path = tuple(pos_to_int[i] for i in path)
            return path
        
        # get current position, possible moves, and moves tried for that position
        current_log = move_logs[-1]
        current_position = current_log["position"]
        possible_moves = current_log["possible_moves"]
        moves_tried_index = current_log["index"]

        if moves_tried_index < len(possible_moves):
            # choose a move if all possible moves from this position haven't been tried
            new_x, new_y = possible_moves[moves_tried_index]
            current_log["index"] += 1

            x, y = new_x, new_y
            move_number += 1
            
            # mark the position as visited
            board[y][x] = move_number
            
            # add position to the path
            path.append((x, y))

            # get next possible moves from the new position
            next_moves = get_moves(x, y, board)
            next_moves = prioritize_moves(next_moves, board)
            
            move_logs.append({
                "position": (x, y),
                "possible_moves": next_moves,
                "index": 0
            })
            
        else: 
            ### backtracking 
            # unmark the current move and remove it from the path 
            move_number -= 1
            board[y][x] = -1
            move_logs.pop()
            path.pop()
            if len(move_logs) > 1:
                # move knight back to the previous square
                x, y = path[-1]
                move_logs[-1]["index"] += 1

    # if the algorithm backtracks all the way to the first position, return None
    return None


def generate_tours(
        total_tours,
        out_queue,
        shared_set,
        shared_set_lock
):
    """

    :param total_tours: The total number of tours to generate
    :param out_queue: Queue from Python's multiprocessing module
    :param shared_set: A dictionary shared across all the different CPUs
    :param shared_set_lock: Lock all processes when adding a new tour
    :return:
    """
    generated = 0

    while generated < total_tours:
        # choosing a random row and column number
        random_x = np.random.randint(0, args.board_length - 1)
        random_y = np.random.randint(0, args.board_height - 1)

        tour = solve_knights_tour_bt(
            initial_coordinate=(random_x, random_y)
        )

        if tour is not None:
            with shared_set_lock:
                if tour not in shared_set:
                    shared_set[tour] = None
                    out_queue.put(tour)
                    generated += 1


def save_as_parquet(
        tours,
        output_directory
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
        f"{args.board_length}x{args.board_height}_"
        f"{args.generate}_"
        f"{current_date_formatted}.parquet"
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


def save_tours_in_batches(
        out_queue,
        batch_size_save=int(1e4)
):
    """

    :param out_queue: Queue from the multiprocessing module
    :param batch_size_save: Batch value to periodically save the tours
    :return: A compressed parquet file with the solved tours
    """
    batch_of_tours = []

    # adding custom progress bar
    # with tqdm(total=args.generate, desc='Tours found') as pbar:
    while True:
        tours_so_far = out_queue.get()

        if tours_so_far is None:
            break

        batch_of_tours.append(tours_so_far)

        # updating progress bar
        # pbar.update(1)

        if len(batch_of_tours) >= batch_size_save:
            save_as_parquet(
                tours=batch_of_tours,
                output_directory=args.savedir
            )

            batch_of_tours = []

    # saving any remaining tours
    # helpful when new tours are generated but the count is less than the batch size
    if len(batch_of_tours) != 0:
        save_as_parquet(
            tours=batch_of_tours,
            output_directory=args.savedir
        )


def main():
    tours_to_generate = args.generate
    number_of_cpus = multiprocessing.cpu_count()

    # defining tours to generate per CPU
    tours_per_cpu = tours_to_generate // number_of_cpus
    extra_tours_cpu = tours_to_generate % number_of_cpus

    batch_size = int(1e4)
    
    # crucial to include the line below for the HPC
    multiprocessing.set_start_method("spawn")
    
    queue = multiprocessing.Queue(maxsize=batch_size)

    # a shared dictionary that is used to track unique tours using keys
    manager = multiprocessing.Manager()
    shared_set = manager.dict()

    if args.existing_tours:
        existing_tours_parquet_rows = pl.read_parquet(args.existing_tours).rows()
        shared_set.update({i: None for i in existing_tours_parquet_rows})

    # deleting file from memory as it won't be necessary anymore
    del existing_tours_parquet_rows
    
    shared_set_lock = multiprocessing.Lock()

    processes = []

    for iteration in range(number_of_cpus):
        # adding extra tours to the last CPU
        if iteration == number_of_cpus - 1:
            tours_per_cpu += extra_tours_cpu

        p = multiprocessing.Process(
            target=generate_tours,
            args=(tours_per_cpu, queue, shared_set, shared_set_lock)
        )

        p.start()
        processes.append(p)

    saver_process = multiprocessing.Process(
        target=save_tours_in_batches,
        args=(
            queue,
        )
    )
    saver_process.start()

    # joining all processes after they finish
    for p in processes:
        p.join()

    # ending the saver process
    queue.put(None)
    saver_process.join()

if __name__ == '__main__':
    main()

