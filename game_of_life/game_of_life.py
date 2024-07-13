"""
https://www.geeksforgeeks.org/program-for-conways-game-of-life/

Initially, there is a grid with some cells which may be alive or dead. Our task is to generate the next generation of cells based on the following rules: 

Any live cell with fewer than two live neighbors dies as if caused by underpopulation.
Any live cell with two or three live neighbors lives on to the next generation.
Any live cell with more than three live neighbors dies, as if by overpopulation.
Any dead cell with exactly three live neighbors becomes a live cell, as if by reproduction.
"""

N_ROWS = 20
N_COLS = 20
DELAY = 0.1

import numpy as np
import sys
import time


def get_next_state(state: np.ndarray) -> np.ndarray:
    n_rows, n_cols = state.shape
    next_state = np.empty((N_ROWS, N_COLS))
    for i in range(n_rows):
        for j in range(n_cols):
            currently_alive = state[i, j]
            num_neighbors = count_alive_neighbors(state, i, j)
            # Any live cell with fewer than two live neighbors dies as if caused by underpopulation.
            # or: Any live cell with more than three live neighbors dies, as if by overpopulation.
            if num_neighbors < 2 or num_neighbors > 3:
                next_state[i, j] = 0
            # Any dead cell with exactly three live neighbors becomes a live cell, as if by reproduction.
            elif num_neighbors == 3:
                next_state[i, j] = 1
            # Any live cell with two or three live neighbors lives on to the next generation.
            else:
                next_state[i, j] = currently_alive
    return next_state


def count_alive_neighbors(state: np.ndarray, row: int, col: int) -> int:
    return state[row - 1 : row + 2, col - 1 : col + 2].sum() - state[row, col]


def set_initial_state(state: np.ndarray) -> None:
    state[5, 5] = 1
    state[6, 5] = 1
    state[5, 6] = 1
    state[5, 7] = 1
    state[4, 6] = 1


def set_blinker(state: np.ndarray) -> None:
    state[5, 5] = 1
    state[5, 6] = 1
    state[5, 4] = 1


def render_state(state: np.ndarray) -> None:
    print(state)


def main() -> None:
    state = np.zeros((N_ROWS, N_COLS))
    set_initial_state(state)
    while True:
        state = get_next_state(state)
        render_state(state)
        time.sleep(DELAY)


if __name__ == "__main__":
    main()
