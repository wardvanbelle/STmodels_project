# Function of this file is to be able to write and test functions line by line
import numpy as np

board_size = 30 # size of board 
num_people = 100 # number of people
exit_locs = ([int(board_size/2-2), int(board_size/2-1), int(board_size/2), int(board_size/2+1)], [0, 0, 0, 0]) # exit locations (middle of left wall)
obstacle_locs = () # No obstacles for now
S_wall = 500
S_exit = 1
mu = 1.5

"""init_S: To be verified
S = np.ones((board_size, board_size)) * np.inf # Initialise array full of +infinity since filling up S is based on selecting minimum values
side_neighbour_mask = np.zeros((3, 3), dtype = bool)
side_neighbour_mask[0, 1] = side_neighbour_mask[1, 0] = side_neighbour_mask[1, -1] = side_neighbour_mask[-1, 1] = True

diag_neighbour_mask = np.zeros((3,3), dtype = bool)
diag_neighbour_mask[0, 0] = diag_neighbour_mask[0, -1] = diag_neighbour_mask[-1, 0] = diag_neighbour_mask[-1, -1] = True

S[exit_locs] = S_exit

curr_cells = [[exit_locs[0][i], exit_locs[1][i]] for i in range(len(exit_locs[0]))]
next_cells = []
done_cells = []


# The exit cells are the only cells on the border to evaluate and require a special treatment (don't select any cells outside of the existing grid)
for y, x in curr_cells:
    for i in np.arange(np.maximum(1, y-1), np.minimum(board_size, y+2)):
        for j in np.arange(np.maximum(1, x-1), np.minimum(board_size, x+2)):
            if (i, j) != (y, x):
                if ((y-i)+(x-j))%2 == 0: # diagonal neighbour
                    S[i,j] = np.minimum(S[y, x] + mu, S[i, j])
                else: # side neighbour
                    S[i,j] = np.minimum(S[y, x] + 1, S[i, j])

                next_cells += [[i, j]]

while next_cells:
    done_cells += curr_cells
    curr_cells = next_cells
    next_cells = []

    for y, x in curr_cells:
        S[y-1:y+2, x-1:x+2][side_neighbour_mask] = np.minimum(S[y, x] + 1, S[y-1:y+2, x-1:x+2][side_neighbour_mask])
        S[y-1:y+2, x-1:x+2][diag_neighbour_mask] = np.minimum(S[y, x] + mu, S[y-1:y+2, x-1:x+2][diag_neighbour_mask])
        next_cells += [[y+i, x+j] for i in range(-1, 2) for j in range(-1, 2) if i != 0 or j != 0 if y+i > 0 and y+i < board_size-1 if x+j > 0 and x+j < board_size-1] # Select all neighbouring cells but not the cell itself
    next_cells = np.unique(next_cells, axis = 0).tolist() # Specify axis or the list of lists will be flattened to 1 list
    next_cells = [cell for cell in next_cells if not cell in done_cells]


S[:, 0] = S[:, -1] = S_wall
S[0, :] = S[-1, :] = S_wall
if obstacle_locs: # => If the list of obstacles isn't empty
    S[obstacle_locs] = S_wall
S[exit_locs] = S_exit
"""

locations = []

for x in range(1,board_size-1):
    for y in range(1,board_size-1):
        locations.append([x,y])