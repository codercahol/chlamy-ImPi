import numpy as np

# save as a config pickle dynamically whenever it's used for an image with the image name tagged

# blank controls -- structure into new object
NUM_ROWS = 16
NUM_COLUMNS = 24
# x_min and x_max set the up/down bounds of the grid
X_MIN = 52 * 2  # 48 * 2 # 50 for a bunch of stuff (384-ctrl WTs)
CELL_WIDTH_X = 43  # 22 * 2
X_MAX = X_MIN + NUM_ROWS * CELL_WIDTH_X
# y_min and y_max set the left/right bounds of the grid
Y_MIN = 35 * 2
# 35 for light curve,
# ^^  22 - for recent blanks test,# 29 for in/out plate test # 33 for 384-ctrl WTs
# when this is not integer, every other cell has more pixels
CELL_WIDTH_Y = 43  # 22 is the right width, with 2x upscaling -> 43
Y_MAX = Y_MIN + NUM_COLUMNS * CELL_WIDTH_Y

# upsample images to avoid half-pixel problem
# controls with light on vs off
# use Adrians old WT data to compare LED exposed to edge strains
## (( investive seg anything model))

# ant:
# - plan abx induction plasmid
# - run simple Flux parity model
# - write up some stuff?
# - find bacteroides metatranscriptomics data -- ask Sophie
