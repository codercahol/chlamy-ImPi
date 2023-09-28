import numpy as np

# maybe define constants by plate type

# 384-well plate (?)
NUM_ROWS = 10
NUM_COLUMNS = 24
# x_min and x_max set the up/down bounds of the grid
X_MIN = 54
CELL_WIDTH_X = 21
X_MAX = X_MIN + NUM_ROWS * CELL_WIDTH_X
# y_min and y_max set the left/right bounds of the grid
Y_MIN = 43
CELL_WIDTH_Y = 21
Y_MAX = Y_MIN + NUM_COLUMNS * CELL_WIDTH_Y

# blank controls -- structure into new object
NUM_ROWS = 10
NUM_COLUMNS = 24
# x_min and x_max set the up/down bounds of the grid
X_MIN = 54
CELL_WIDTH_X = 21
X_MAX = X_MIN + NUM_ROWS * CELL_WIDTH_X
# y_min and y_max set the left/right bounds of the grid
Y_MIN = 43
CELL_WIDTH_Y = 21
Y_MAX = Y_MIN + NUM_COLUMNS * CELL_WIDTH_Y
