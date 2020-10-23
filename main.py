import random
from sympy import *
from typing import List, Tuple
from queue import PriorityQueue


class Minesweeper:
    """
    Defines Minesweeper class, which simulates Minesweeper game.
    """

    def __init__(self, d, n):
        """
        Initialize Minesweeper properties.
        Args:
            d: dimension of board
            n: number of mines
        """
        self._d = d
        self._n = n
        self.environment = self.make_board()
        self.agent_board = [[-1 for _ in range(self._d)] for _ in range(self._d)]

    def make_board(self) -> List[List[List[int]]]:
        """
        Initialize board. At each index of the board, we use a list of size 4. The list is: [# of mines around, # of
        safe neighbors identified, # of mine neighbors identified, # of hidden squares]. If there is a mine at an index,
        it is represented as [9, 0, 0, 8].
        Returns:
            List[List[List[int]]]: 2D board representing Minesweeper board

        """
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
        """
        Print board.
        """
        for row in range(self._d):
            print(self.environment[row])

    def print_agent_board(self) -> None:
        """
        Print board from perspective of agent.
        """
        for row in range(self._d):
            print(self.agent_board[row])

    def play_game(self, game_type: str) -> None:
        """
        Driver for  agent game, basic and improved
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
                    if game_type == 'basic':
                        visited = [[False for _ in range(self._d)] for _ in range(self._d)]
                        self.update_queues_island(curr_cell, safe_cell_queue, mine_cell_queue, visited)
                    elif game_type == 'improved':
                        self.update_knowledge_base(safe_cell_queue, mine_cell_queue)
                    # self._update_queues(curr_cell, safe_cell_queue, mine_cell_queue)

                # Picked a mine
                else:
                    mines_exploded += 1

                    # Update mine on agent board
                    self.agent_board[curr_cell[0]][curr_cell[1]] = 9

                    # Update environment
                    self.update_environment(curr_cell, 'mine')

                    print("Exploded a mine.")

                    # if game_type == 'basic':
                    #     visited = [[False for _ in range(self._d)] for _ in range(self._d)]
                    #     self.update_queues_island(curr_cell, safe_cell_queue, mine_cell_queue, visited)
                    if game_type == 'improved':
                        self.update_knowledge_base(safe_cell_queue, mine_cell_queue)

                revealed_cells += 1

                # self.print_environment()
                # self.print_agent_board()
                # print("Safe Cell Queue: ", safe_cell_queue)
                # print("Mine Queue: ", mine_cell_queue)
                # print("Revealed cells: ", revealed_cells)
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

                        if game_type == 'improved':
                            self.update_knowledge_base(safe_cell_queue, mine_cell_queue)

                        # self.print_environment()
                        # self.print_agent_board()
                        # print("Safe Cell Queue: ", safe_cell_queue)
                        # print("Mine Queue: ", mine_cell_queue)
                        # print("Revealed cells: ", revealed_cells)
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
                        if game_type == 'basic':
                            visited = [[False for _ in range(self._d)] for _ in range(self._d)]
                            self.update_queues_island(curr_cell, safe_cell_queue, mine_cell_queue, visited)
                        elif game_type == 'improved':
                            self.update_knowledge_base(safe_cell_queue, mine_cell_queue)

                        revealed_cells += 1

                        # self.print_environment()
                        # self.print_agent_board()
                        # print("Safe Cell Queue: ", safe_cell_queue)
                        # print("Mine Queue: ", mine_cell_queue)
                        # print("Revealed cells: ", revealed_cells)
                        print()

        # Game is over
        print("Game over")
        print("Mines safely identified: ", self._n - mines_exploded)
        print("Total mines: ", self._n)
        print("Revealed cells: ", revealed_cells)
        self.print_agent_board()

    def pick_random_cell(self) -> Tuple[int, int]:
        """
        Pick a random cell in the board that is hidden.
        """
        i = random.randint(0, self._d - 1)
        j = random.randint(0, self._d - 1)
        while self.agent_board[i][j] != -1:
            i = random.randint(0, self._d - 1)
            j = random.randint(0, self._d - 1)

        return i, j

    def update_environment(self, curr_cell: Tuple[int, int], flag: str) -> None:
        """
        Update environment values based on newly revealed cell, and whether ce
        Args:
            curr_cell: cell that was revelead
            flag: whether revealed cell was safe or mine

        Returns:

        """
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

    def update_knowledge_base(self, safe_cell_queue: List[Tuple[int, int]],
                              mine_cell_queue: List[Tuple[int, int]]):
        """
        Generate equations for the knowledge base in matrix and solve as much as possible.
        Args:
            safe_cell_queue: queue of safe cells
            mine_cell_queue: queue of mine cells

        Returns:

        """
        print('Updating knowledge base...')
        matrix_list = []
        for i in range(self._d):
            for j in range(self._d):
                # Safe / clue cell
                if self.agent_board[i][j] in range(0, 9):
                    # index_to_add = i * self._d + j
                    # safe_cell_equation = []
                    #
                    # # Add equation for safe cell
                    # for k in range(self._d ** 2):
                    #     if k == index_to_add:
                    #         safe_cell_equation.append(1)
                    #     else:
                    #         safe_cell_equation.append(0)
                    # safe_cell_equation.append(0)

                    # Add equation for safe cell clue
                    clue_equation = self.generate_neighbors_equation(i, j)
                    hidden_mines = self.environment[i][j][0] - self.environment[i][j][2]
                    clue_equation.append(hidden_mines)

                    # matrix.append(safe_cell_equation)
                    matrix_list.append(clue_equation)

        matrix = Matrix(matrix_list)
        rref_matrix = matrix.rref()[0]
        reduced = False
        # pprint(matrix)
        # pprint(rref_matrix)

        while not reduced:
            reduced = True
            for i in range(rref_matrix.shape[0]):
                var_num_per_equation = []

                for j in range(rref_matrix.shape[1] - 1):
                    if rref_matrix[i, j] != 0:
                        var_num_per_equation.append(j)

                print('Var num per equation: ', var_num_per_equation)

                sum_in_var_nums = 0
                for index in var_num_per_equation:
                    sum_in_var_nums += rref_matrix[i, index]
                # All safe cells in equation
                if rref_matrix[i, rref_matrix.shape[1] - 1] == 0 and all(rref_matrix[i, index] > 0 for index in
                                                                         var_num_per_equation):
                    print('All safe equation')
                    for element in var_num_per_equation:
                        j = int(element % self._d)
                        i = int((element - j) / self._d)
                        cell = (i, j)
                        if cell not in safe_cell_queue:
                            safe_cell_queue.append(cell)
                        self.update_rref_matrix(rref_matrix, i, j, 0)

                # All mine cells in equation
                elif rref_matrix[i, rref_matrix.shape[1] - 1] == sum_in_var_nums and all(rref_matrix[i, index] > 0 for index in var_num_per_equation):
                    print('All mine equation')
                    for element in var_num_per_equation:
                        j = int(element % self._d)
                        i = int((element - j) / self._d)
                        cell = (i, j)
                        if cell not in mine_cell_queue:
                            mine_cell_queue.append(cell)
                        self.update_rref_matrix(rref_matrix, i, j, 1)

                # 2 vars in equation
                elif len(var_num_per_equation) == 2:
                    print('2 var equation')
                    j_0 = int(var_num_per_equation[0] % self._d)
                    i_0 = int((var_num_per_equation[0] - j_0) / self._d)
                    j_1 = int(var_num_per_equation[1] % self._d)
                    i_1 = int((var_num_per_equation[1] - j_1) / self._d)
                    first_cell = (i_0, j_0)
                    second_cell = (i_1, j_1)

                    # Guess and check the 4 combos of 0,0 0,1 1,0 1,1
                    # if rref_matrix[i][-1] == 0:
                    #     continue
                    solution_flag = 0
                    coefficient_0 = rref_matrix[i, var_num_per_equation[0]]
                    coefficient_1 = rref_matrix[i, var_num_per_equation[1]]
                    if coefficient_0 * 0 + coefficient_1 * 1 == rref_matrix[i, -1]:
                        solution_flag = 1
                    if coefficient_0 * 0 + coefficient_1 * 0 == rref_matrix[i, -1]:
                        if solution_flag != 0:
                            continue
                        else:
                            solution_flag = 2
                    if coefficient_0 * 1 + coefficient_1 * 0 == rref_matrix[i, -1]:
                        if solution_flag != 0:
                            continue
                        else:
                            solution_flag = 3
                    if coefficient_0 * 1 + coefficient_1 * 1 == rref_matrix[i, -1]:
                        if solution_flag != 0:
                            continue
                        else:
                            solution_flag = 4

                    # Checking
                    if solution_flag == 1:
                        if first_cell not in safe_cell_queue:
                            safe_cell_queue.append(first_cell)
                        if second_cell not in mine_cell_queue:
                            mine_cell_queue.append(second_cell)
                        self.update_rref_matrix(rref_matrix, i_0, j_0, 0)
                        self.update_rref_matrix(rref_matrix, i_1, j_1, 1)
                        reduced = False
                    elif solution_flag == 2:
                        if first_cell not in safe_cell_queue:
                            safe_cell_queue.append(first_cell)
                        if second_cell not in safe_cell_queue:
                            safe_cell_queue.append(second_cell)
                        self.update_rref_matrix(rref_matrix, i_0, j_0, 0)
                        self.update_rref_matrix(rref_matrix, i_1, j_1, 0)
                        reduced = False
                    elif solution_flag == 3:
                        if first_cell not in mine_cell_queue:
                            mine_cell_queue.append(first_cell)
                        if second_cell not in safe_cell_queue:
                            safe_cell_queue.append(second_cell)
                        self.update_rref_matrix(rref_matrix, i_0, j_0, 1)
                        self.update_rref_matrix(rref_matrix, i_1, j_1, 0)
                        reduced = False
                    elif solution_flag == 4:
                        if first_cell not in mine_cell_queue:
                            mine_cell_queue.append(first_cell)
                        if second_cell not in mine_cell_queue:
                            mine_cell_queue.append(second_cell)
                        self.update_rref_matrix(rref_matrix, i_0, j_0, 1)
                        self.update_rref_matrix(rref_matrix, i_1, j_1, 1)
                        reduced = False

                # 1 var in equation
                elif len(var_num_per_equation) == 1:
                    print('1 var equation')
                    j = int(var_num_per_equation[0] % self._d)
                    i = int((var_num_per_equation[0] - j) / self._d)
                    cell = (i, j)

                    # Mine
                    if rref_matrix[i, rref_matrix.shape[1]-1] == 1:
                        if cell not in mine_cell_queue:
                            mine_cell_queue.append(cell)
                    # Safe
                    else:
                        if cell not in safe_cell_queue:
                            safe_cell_queue.append(cell)
        print('Finished checking knowledge base.')

    def update_rref_matrix(self, rref_matrix: Matrix, i: int, j: int, val: int):
        """
        Using the solved variable, expressed as the "j" in the input, plug into all other equations in knowledge base
        that contain that variable
        Args:
            rref_matrix: knowledge base
            i: equation number where the variable was just solved
            j: variable number
            val: value of variable (0 or 1)

        Returns:

        """
        var_num = int(i * self._d + j)
        print('Var_num:')
        print(var_num)

        for x in range(rref_matrix.shape[0]):
            coefficient = rref_matrix[x, var_num]
            rref_matrix[x, rref_matrix.shape[1] - 1] -= coefficient * val
            rref_matrix[x, var_num] = 0

    def generate_neighbors_equation(self, i: int, j: int) -> List:
        """
        Generate equation of 0s and 1s based on whether neighbors to the given cell are hidden. 1s represent a hidden
        cell, 0s represent non-neighbors (must be zeroed out for the matrix equation).
        Args:
            i: x index of cell
            j: y index of cel

        Returns: equation of neighbors

        """
        equation = []
        for x in range(self._d):
            for y in range(self._d):
                if abs(i - x) <= 1 and abs(j - y) <= 1 and (x, y) != (i, j) and self.agent_board[x][y] == -1:
                    equation.append(1)
                else:
                    equation.append(0)

        # for x in range(-1, 2):
        #     for y in range(-1, 2):
        #         if x != 0 or y != 0:
        #             if 0 <= i + x < self._d and 0 <= j + y < self._d:
        #                 # Add hidden neighbor cell to equation
        #                 if self.agent_board[x + i][y + j] == -1:
        #                     equation.append(1)
        #                 else:
        #                     equation.append(0)
        return equation

    def update_queues_island(self, curr_cell: Tuple[int, int], safe_cell_queue: List[Tuple[int, int]],
                             mine_cell_queue: List[Tuple[int, int]], visited: List[List[bool]]) -> None:
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

    def _update_queues(self, curr_cell: Tuple[int, int], safe_cell_queue: List[Tuple[int, int]],
                       mine_cell_queue: List[Tuple[int, int]]) -> None:
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
        """
        Calculate max possible neighbors based on whether cell is corner, edge, or neither.
        Args:
            i: x dimension index of cell
            j: y dimension index of cell

        Returns:
            int: possible number of neighbors
        """
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

    minesweeper.play_game('improved')


if __name__ == '__main__':
    main(10, 40)
