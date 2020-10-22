import random
from typing import List, Tuple
from queue import PriorityQueue


class Minesweeper:

    def __init__(self, d, n):
        self._d = d
        self._n = n
        self.environment = self.make_board()
        self.agent_board = [[-1 for _ in range(self._d)] for _ in range(self._d)]

    def make_board(self) -> List[List[List[int]]]:
        board = [[[0, 0, 0, 8] for _ in range(self._d)] for _ in range(self._d)]

        # Randomly places the mines
        for _ in range(self._n):
            i = random.randint(0, self._d - 1)
            j = random.randint(0, self._d - 1)
            while board[i][j] == [9, 0, 0, 8]:
                i = random.randint(0, self._d - 1)
                j = random.randint(0, self._d - 1)
            board[i][j] = [9, 0, 0, 8]

            # Updates the number of mines around for safe cells
            for x in range(-1, 2):
                for y in range(-1, 2):
                    if x != 0 or y != 0:
                        if 0 <= i + x < self._d and 0 <= j + y < self._d and board[i + x][j + y] != [9, 0, 0, 8]:
                            board[i + x][j + y][0] += 1

        # Update hidden neighbor counts for edges
        for x in range(self._d):
            for y in range(self._d):
                board[x][y][3] = self.get_possible_neighbors(x, y)
                # if x == 0 or x == self._d-1:
                #     if y == 0 or y == self._d-1:
                #         # Corner
                #         board[x][y][3] = 3
                #     else:
                #         # Edge
                #         board[x][y][3] = 5
                #
                # elif y == 0 or y == self._d-1:
                #     # Edge
                #     board[x][y][3] = 5

        return board

    def print_environment(self) -> None:
        for row in range(self._d):
            print(self.environment[row])

    def print_agent_board(self) -> None:
        for row in range(self._d):
            print(self.agent_board[row])

    def play_basic_game(self) -> None:
        """
        Driver for basic agent game, looking at essentialy 1 clue at a time.
        """
        print("Let's play!")

        revealed_cells = 0
        mines_exploded = 0

        # Create queues
        safe_cell_queue: List[Tuple[int, int]] = []
        mine_cell_queue: List[Tuple[int, int]] = []

        # Play until all cells are revealed
        while revealed_cells < self._d * self._d:

            # Pick a random cell if no hidden cell can be conclusively identified
            if not safe_cell_queue and not mine_cell_queue:

                curr_cell = self.pick_random_cell()

                print("Current random cell: ", curr_cell)

                # Picked a safe cell
                if self.environment[curr_cell[0]][curr_cell[1]][0] != 9:
                    # Update clue on agent board
                    self.agent_board[curr_cell[0]][curr_cell[1]] = self.environment[curr_cell[0]][curr_cell[1]][0]

                    # Update environment
                    self.update_environment(curr_cell, 'safe')

                    # Update queues for all revealed cells on the island
                    visited = [[False for _ in range(self._d)] for _ in range(self._d)]
                    self.update_queues_island(curr_cell, safe_cell_queue, mine_cell_queue, visited)
                    # self._update_queues(curr_cell, safe_cell_queue, mine_cell_queue)

                # Picked a mine
                else:
                    mines_exploded += 1

                    # Update mine on agent board
                    self.agent_board[curr_cell[0]][curr_cell[1]] = 9

                    # Update environment
                    self.update_environment(curr_cell, 'mine')

                    print("Exploded a mine.")

                revealed_cells += 1

                self.print_environment()
                self.print_agent_board()
                print("Safe Cell Queue: ", safe_cell_queue)
                print("Mine Queue: ", mine_cell_queue)
                print("Revealed cells: ", revealed_cells)
                print()

            # Cells have been identified as mine/safe, so reveal/mark and update info
            else:
                # (i am not sure if order of reveal matters)
                while mine_cell_queue or safe_cell_queue:

                    # Reveal the mines
                    while mine_cell_queue:
                        curr_cell = mine_cell_queue.pop()

                        print("Current mine cell: ", curr_cell)

                        # Update mine on agent board
                        self.agent_board[curr_cell[0]][curr_cell[1]] = 9

                        # Update environment
                        self.update_environment(curr_cell, 'mine')

                        revealed_cells += 1

                        self.print_environment()
                        self.print_agent_board()
                        print("Safe Cell Queue: ", safe_cell_queue)
                        print("Mine Queue: ", mine_cell_queue)
                        print("Revealed cells: ", revealed_cells)
                        print()

                    # Reveal the safe cells
                    while safe_cell_queue:
                        curr_cell = safe_cell_queue.pop()

                        print("Current safe cell: ", curr_cell)

                        # Update clue on agent board
                        self.agent_board[curr_cell[0]][curr_cell[1]] = self.environment[curr_cell[0]][curr_cell[1]][0]

                        # Update environment
                        self.update_environment(curr_cell, 'safe')

                        # Update queues
                        visited = [[False for _ in range(self._d)] for _ in range(self._d)]
                        self.update_queues_island(curr_cell, safe_cell_queue, mine_cell_queue, visited)

                        revealed_cells += 1

                        self.print_environment()
                        self.print_agent_board()
                        print("Safe Cell Queue: ", safe_cell_queue)
                        print("Mine Queue: ", mine_cell_queue)
                        print("Revealed cells: ", revealed_cells)
                        print()

        # Game is over
        print("Game over")
        print("Mines safely identified: ", self._n - mines_exploded)
        print("Total mines: ", self._n)
        print("Revealed cells: ", revealed_cells)
        self.print_agent_board()

    def pick_random_cell(self) -> Tuple[int, int]:
        i = random.randint(0, self._d - 1)
        j = random.randint(0, self._d - 1)
        while self.agent_board[i][j] != -1:
            i = random.randint(0, self._d - 1)
            j = random.randint(0, self._d - 1)

        return i, j

    def update_environment(self, curr_cell: Tuple[int, int], flag: str) -> None:
        i = curr_cell[0]
        j = curr_cell[1]

        # Update curr_cell's neighbor values
        for x in range(-1, 2):
            for y in range(-1, 2):
                if x != 0 or y != 0:
                    if 0 <= i + x < self._d and 0 <= j + y < self._d:

                        if flag == 'safe':
                            # Update number of safe cells identified
                            self.environment[i + x][j + y][1] += 1
                        else:
                            # Update number of mines identified
                            self.environment[i + x][j + y][2] += 1

                        # Update number of hidden squares
                        self.environment[i + x][j + y][3] -= 1

    def update_queues_island(self, curr_cell: Tuple[int, int], safe_cell_queue: List[Tuple[int, int]], mine_cell_queue: List[Tuple[int, int]], visited: List[List[bool]]) -> None:
        """
        Check for whether every hidden neighbor is safe / a mine for all revealed cells on the island. Note: the island
        is defined as all connected cells such that they are adjacent directly and diagonally.
        Args:
            curr_cell: current cell
            safe_cell_queue: queue of cells which are determined to be definitely safe
            mine_cell_queue: queue of cells which are determined to be definitely mines
            visited: marks cells as visited while update queues on the island
        """
        i = curr_cell[0]
        j = curr_cell[1]
        if i < 0 or i == self._d or j < 0 or j == self._d or visited[i][j] or self.agent_board[i][j] == -1:
            return

        print('Checking island...')
        print(i)
        print(j)

        visited[i][j] = True
        self._update_queues(curr_cell, safe_cell_queue, mine_cell_queue)

        self.update_queues_island((i + 1, j), safe_cell_queue, mine_cell_queue, visited)
        self.update_queues_island((i - 1, j), safe_cell_queue, mine_cell_queue, visited)
        self.update_queues_island((i, j + 1), safe_cell_queue, mine_cell_queue, visited)
        self.update_queues_island((i, j - 1), safe_cell_queue, mine_cell_queue, visited)

        # Not sure if these 4 should also recurse / count as the island, but might as well (minimal cost for now,
        # only has benefit)
        self.update_queues_island((i - 1, j - 1), safe_cell_queue, mine_cell_queue, visited)
        self.update_queues_island((i + 1, j + 1), safe_cell_queue, mine_cell_queue, visited)
        self.update_queues_island((i - 1, j + 1), safe_cell_queue, mine_cell_queue, visited)
        self.update_queues_island((i + 1, j - 1), safe_cell_queue, mine_cell_queue, visited)

    def _update_queues(self, curr_cell: Tuple[int, int], safe_cell_queue: List[Tuple[int, int]], mine_cell_queue: List[Tuple[int, int]]) -> None:
        """
        Checks if for the current cell, the hidden neighbors can be determined to be definitely safe or definitely mines.
        If either, add the corresponding cells to the corresponding safe or mine cell queue.
        Args:
            curr_cell: current cell
            safe_cell_queue: queue of cells which are determined to be definitely safe
            mine_cell_queue: queue of cells which are determined to be definitely mines
        """
        i = curr_cell[0]
        j = curr_cell[1]

        possible_neighbors = self.get_possible_neighbors(i, j)
        neighboring_mines = self.environment[i][j][0]
        neighboring_safe_cells = possible_neighbors - neighboring_mines
        neighboring_safe_cells_revealed = self.environment[i][j][1]
        neighboring_mines_revealed = self.environment[i][j][2]
        neighboring_hidden_squares = self.environment[i][j][3]

        if neighboring_mines - neighboring_mines_revealed == neighboring_hidden_squares:
            # Every hidden neighbor is a mine, so add all to mine_cell_queue
            for x in range(-1, 2):
                for y in range(-1, 2):
                    if x != 0 or y != 0:
                        if 0 <= i + x < self._d and 0 <= j + y < self._d and self.agent_board[i + x][j + y] == -1:
                            mine_cell = i + x, j + y
                            # Make sure it does not already exist in either queue
                            if mine_cell not in mine_cell_queue and mine_cell not in safe_cell_queue:
                                mine_cell_queue.append(mine_cell)

        elif neighboring_safe_cells - neighboring_safe_cells_revealed == neighboring_hidden_squares:
            # Every hidden neighbor is a safe cell, so add all to safe_cell_queue
            for x in range(-1, 2):
                for y in range(-1, 2):
                    if x != 0 or y != 0:
                        if 0 <= i + x < self._d and 0 <= j + y < self._d and self.agent_board[i + x][j + y] == -1:
                            safe_cell = i + x, j + y
                            # Make sure it does not already exist in either queue
                            if safe_cell not in mine_cell_queue and safe_cell not in safe_cell_queue:
                                safe_cell_queue.append(safe_cell)

    def get_possible_neighbors(self, i: int, j: int) -> int:

        if i == 0 or i == self._d - 1:
            if j == 0 or j == self._d - 1:
                # Corner
                possible_neighbors = 3
            else:
                # Edge
                possible_neighbors = 5

        elif j == 0 or j == self._d - 1:
            # Edge
            possible_neighbors = 5
        else:
            # Central cell
            possible_neighbors = 8

        return possible_neighbors


def main(d: int, n: int):
    print('Hello world')
    minesweeper = Minesweeper(d, n)
    minesweeper.print_environment()
    minesweeper.print_agent_board()

    minesweeper.play_basic_game()


if __name__ == '__main__':
    main(3, 2)
