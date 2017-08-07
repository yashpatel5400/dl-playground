"""
sudoku solver (simple)
"""

import copy

def extract_rows_columns_and_boxes(board):
    rows = [set(row) for row in board]
    enum_rows = dict(enumerate(rows))

    columns = [set([row[i] for row in board]) for i in range(9)]
    enum_columns = dict(enumerate(columns))

    enum_boxes = {}
    for i in range(3):
        for j in range(3):
            new_box = []
            for k in range(3):
                new_box.append(board[i * 3][j * 3 + k])
                new_box.append(board[i * 3 + 1][j * 3 + k])
                new_box.append(board[i * 3 + 2][j * 3 + k])
            enum_boxes[i * 3 + j] = set(new_box)

    return enum_rows, enum_columns, enum_boxes

def solve_puzzle_helper(rows, columns, boxes, cur_row, cur_col, cur_board):
    next_col = cur_col + 1
    next_row = cur_row
    if cur_col == 8:
        next_col = 0
        next_row += 1

    if next_row == 9:
        return cur_board

    if cur_board[cur_row][cur_col] != 0:
        possible_solution = solve_puzzle_helper(rows, columns, boxes, next_row, next_col, cur_board)
        if possible_solution is not None:
            return possible_solution

    cur_box = int(cur_row / 3) * 3 + int(cur_col / 3)
    cur_row_entries = rows[cur_row]
    cur_col_entries = columns[cur_col]
    cur_box_entries = boxes[cur_box]

    all_possible = {x for x in range(1, 10)}
    ignored = cur_row_entries.union(cur_col_entries).union(cur_box_entries)
    possible_values = all_possible.difference(ignored)
    for possible_value in possible_values:
        new_rows = copy.deepcopy(rows)
        new_columns = copy.deepcopy(columns)
        new_boxes = copy.deepcopy(boxes)

        new_rows[cur_row].add(possible_value)
        new_columns[cur_col].add(possible_value)
        new_boxes[cur_box].add(possible_value)

        cur_board[cur_row][cur_col] = possible_value
        possibly_solved_board = solve_puzzle_helper(new_rows, new_columns, new_boxes,
                                                    next_row, next_col, cur_board)
        if possibly_solved_board is not None:
            return possibly_solved_board
    return None

def solve_puzzle(board):
    rows, columns, boxes = extract_rows_columns_and_boxes(board)
    solved = solve_puzzle_helper(rows, columns, boxes, 0, 0, board)
    return solved

# 0 is used to identically embed an empty space in the board
board = [
    [0,0,0,8,0,0,0,0,0],
    [4,0,0,0,1,5,0,3,0],
    [0,2,9,0,4,0,5,1,8],

    [0,4,0,0,0,0,1,2,0],
    [0,0,0,6,0,2,0,0,0],
    [0,3,2,0,0,0,0,9,0],

    [6,9,3,0,5,0,8,7,0],
    [0,5,0,4,8,0,0,0,1],
    [0,0,0,0,0,3,0,0,0]
]
solution = solve_puzzle(board)
for row in solution:
    print(row)