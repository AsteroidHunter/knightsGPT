"""
@author: akash
"""

import os
import argparse
import numpy as np
import polars as pl
import multiprocessing
from datetime import date

# np.random.seed(2141341)

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
    required=True
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


def solve_knights_tour(
        initial_coordinate
):
    """

    :param initial_coordinate: Tuple with row and column number
    :return: A completed tour as a 1-D sequence; None if the tour fails
    """

    board = [
        [-1 for _ in range(args.board_length)] for _ in range(args.board_height)
    ]

    move_x, move_y = initial_coordinate

    board[move_y][move_x] = 0

    for move_number in range(1, args.board_length * args.board_height):
        possible_moves = []

        for x, y in possible_steps_offset.values():
            new_x, new_y = move_x + x, move_y + y

            if valid_move(new_x, new_y, board):
                move_degree = degree_of_the_move(new_x, new_y, board)
                possible_moves.append([move_degree, new_x, new_y])

        # terminating if the knight faces a dead end
        if not possible_moves:
            return None

        # applying the Warnsdorff heuristic
        smallest_degree = min(possible_moves)[0]
        best_moves = [
            (i, j) for move_count, i, j in possible_moves if move_count == smallest_degree
        ]

        # if there are several possible moves w the same degree, choosing one at random
        move_x, move_y = best_moves[np.random.choice(len(best_moves))]

        # marking move done
        board[move_y][move_x] = move_number

    # retrieving the solution in the form of a sequence of index values
    solution = tuple(i.item() for i in np.argsort(np.ravel(board)))

    return solution


def generate_tours(
        total_tours,
        out_queue,
        shared_set,
        shared_set_lock
):
    """

    :param total_tours:
    :param out_queue:
    :param shared_set:
    :param shared_set_lock:
    :return:
    """
    # crucial to have this here for at least 90% uniqueness in generated solutions
    # also, make sure largest seed > number of solutions to generate
    np.random.seed(np.random.choice(int(1e15)))

    generated = 0

    while generated < total_tours:
        # choosing a random row and column number
        random_x = np.random.randint(0, args.board_length - 1)
        random_y = np.random.randint(0, args.board_height - 1)

        tour = solve_knights_tour(
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

    while True:
        tours_so_far = out_queue.get()

        if tours_so_far is None:
            break

        batch_of_tours.append(tours_so_far)

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
    number_of_cpus = 20 # multiprocessing.cpu_count()
    
    # defining tours to generate per CPU
    tours_per_cpu = tours_to_generate // number_of_cpus
    extra_tours_cpu = tours_to_generate % number_of_cpus
    
    # print("Number of CPUs:", number_of_cpus)
    
    batch_size = int(1e4)
    multiprocessing.set_start_method("spawn")
    queue = multiprocessing.Queue(maxsize=batch_size)

    # a shared dictionary that is used to track unique tours using keys
    manager = multiprocessing.Manager()
    shared_set = manager.dict()
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
