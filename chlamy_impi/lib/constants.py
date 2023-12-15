import numpy as np


class Config:
    def __init__(self):
        # Default settings for a 384-well plate
        self.NUM_ROWS = 16
        self.NUM_COLUMNS = 24
        # x_min and x_max set the up/down bounds of the grid
        self.X_MIN = 50 * 2
        self.CELL_WIDTH_X = 43  # 22 * 2
        self.X_MAX = self.X_MIN + self.NUM_ROWS * self.CELL_WIDTH_X
        # y_min and y_max set the left/right bounds of the grid
        self.Y_MIN = 35 * 2
        self.CELL_WIDTH_Y = 43  # 22 is the right width, with 2x upscaling -> 43
        self.Y_MAX = self.Y_MIN + self.NUM_COLUMNS * self.CELL_WIDTH_Y

    def alter(setting):
        if setting == "test_plate":
            self.NUM_ROWS = 10
            self.X_MIN = 54
            self.CELL_WIDTH_X = 21
            self.X_MAX = self.X_MIN + self.NUM_ROWS * self.CELL_WIDTH_X
            self.Y_MIN = 43
            self.CELL_WIDTH_Y = 21
            self.Y_MAX = self.Y_MIN + self.NUM_COLUMNS * self.CELL_WIDTH_Y
        # add other elif's if more settings are needed
