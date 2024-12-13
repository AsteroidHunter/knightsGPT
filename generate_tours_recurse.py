import os
import random
import argparse
import polars as pl
import multiprocessing


# setting arguments that could be passed when running the script
parser = argparse.ArgumentParser()
parser.add_argument(
    '--generate',
    type=int,
    help="Number of Knight's Tour solutions to generate",
    required=True
)

parser.add_argument(
    '--num1',
    type=int,
    required=True
)

parser.add_argument(
    '--num2',
    type=int,
    required=True
)

args = parser.parse_args()


pos_to_int = {i: j for i, j in zip(
    [(i, j) for i in range(8) for j in range(8)],
    range(64)
)}

# Define knight moves
possible_steps_offset = [
    (2, 1), (1, 2), (-1, 2), (-2, 1),
    (-2, -1), (-1, -2), (1, -2), (2, -1)
]

def valid_move(x, y, board):
    return 0 <= x < 8 and 0 <= y < 8 and board[y][x] == -1

def degree_of_move(x, y, board):
    count = 0
    for dx, dy in possible_steps_offset:
        if valid_move(x + dx, y + dy, board):
            count += 1
    return count

def make_moves(x, y, board):
    moves = []
    for dx, dy in possible_steps_offset:
        new_x, new_y = x + dx, y + dy
        if valid_move(new_x, new_y, board):
            moves.append((new_x, new_y))
    # Apply degree heuristic with randomness
    moves.sort(key=lambda move: degree_of_move(move[0], move[1], board))
    random.shuffle(moves[:2])  # Introduce randomness in top 2 moves
    return moves

def kt(initial_position, solutions_limit):
    board = [[-1 for _ in range(8)] for _ in range(8)]
    x, y = initial_position
    move_number = 0
    board[y][x] = move_number
    path = [(x, y)]
    solutions = []

    def backtrack(x, y, move_number):
        if move_number == 63:
            solutions.append(tuple(path))
            if len(solutions) >= solutions_limit:
                return True
            return False

        for new_x, new_y in make_moves(x, y, board):
            board[new_y][new_x] = move_number + 1
            path.append((new_x, new_y))
            if backtrack(new_x, new_y, move_number + 1):
                return True
            board[new_y][new_x] = -1
            path.pop()
        return False

    backtrack(x, y, move_number)
    return tuple(tuple(pos_to_int[i] for i in solutions[j]) for j in range(len(solutions)))


def worker(position, solutions_limit, output_queue):
    # Generate knight's tours starting from the given position
    tours = kt(position, solutions_limit)
    output_queue.put(tours)  # Add the solutions to the shared queue


def save_as_parquet(tours, output_directory="./data/"):
    """
    Save the tours to a Parquet file.
    """
    total_moves = len(tours[0])

    # int8 schema to reduce file size
    as_df = pl.DataFrame(
        tours,
        schema={f"move_{i}": pl.Int8 for i in range(total_moves)},
        orient="row"
    )

    file_path = (
        f"{output_directory}"
        f"/tours_"
        f"{8}x{8}_"
        f"{len(tours)}_"
        f"{20241201}_recursive_01_to_21.parquet"
    )

    try:
        # If the file already exists, concatenate new results and save
        if os.path.exists(file_path):
            existing_df = pl.read_parquet(file_path)
            as_df = pl.concat([existing_df, as_df])

        as_df.write_parquet(
            file_path,
            compression="zstd"
        )

    except Exception as e:
        print(f"File could not be saved due to the following error: {e}")

def main():
    pos_to_sample = [
        (0, 1), (0, 2), (0, 3),
        (1, 0), (1, 1), (1, 2), (1, 3),
        (2, 0), (2, 1), (2, 2), (2, 3),
        (3, 0), (3, 1), (3, 2), (3, 3)
    ]

    tours_to_generate = args.generate
    pos_to_sample = pos_to_sample[args.num1:args.num2]
    num_processes = len(pos_to_sample)

    # Shared queue to collect results
    output_queue = multiprocessing.Queue()

    # Start processes
    processes = []
    for position in pos_to_sample:
        p = multiprocessing.Process(
            target=worker,
            args=(position, tours_to_generate, output_queue)
        )
        p.start()
        processes.append(p)

    # Signal end of processing by adding sentinel values
    for p in processes:
        p.join()

    print("All processes have completed.")

    # Collect results and save incrementally
    batch_size = 100_000
    batch_tours = []
    processed_batches = 0

    while True:
        try:
            tours = output_queue.get(timeout=5)  # Wait for results with a timeout
            batch_tours.extend(tours)
            if len(batch_tours) >= batch_size:
                save_as_parquet(batch_tours)
                print(f"Saved batch {processed_batches + 1} of size {len(batch_tours)}")
                batch_tours = []
                processed_batches += 1
        except multiprocessing.queues.Empty:
            # Break if queue is empty and all processes are joined
            break

    # Save any remaining tours
    if batch_tours:
        save_as_parquet(batch_tours)
        print(f"Saved final batch of size {len(batch_tours)}")

    print("All tours have been saved.")


if __name__ == "__main__":
    main()
