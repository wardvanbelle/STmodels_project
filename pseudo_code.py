# step 1: Populate board with random amount of people in random positions

def init_board(r,x,e):
        # populate board with size r x r with x people
        # decide where exits are with e num of exits
        # set remaining walls to 500
    return board 

stampede = True

# cycle

def calc_S(board):
        # calculate the static board based on previous board
    return S 

def calc_D(board):
        # calculate the dynamic board based on previous board
    return D 

def calc_F(board):
        # calculate the event floor board based on previous board
    return F 

while stampede:

    board = cellular_stampede[-1,:,:]

    # step 2: In each time step, calculate the static floor field Sij, the dynamic floor field Dij and the event floor field Fij with help of the formulas.
    
    S = calc_S(board)
    D = calc_D(board)
    F = calc_F(board)

    # step 3 and 4: For each pedestrian, determine his/her state and make a decision regarding movement behavior. If the pedestrian is in 
    # state Ue,Un or Ae, calculate the transition probability according to Eqs. (15)–(17), respectively. If the pedestrian is in state An, he/she chooses one of the strategies S1, S2 or S3 to evacuate. 
    # If the pedestrian is in state C, calculate the probability of him/her getting up again.
    # Each pedestrian moves to his/her target cell. If the pedestrian chooses a target cell that is occupied by a fallen pedestrian, he/she will be tripped and become a new fallen pedestrian.

    states = {500:'wall', 1:'..'}

    person_pos = board[...]

    next_board = board

    for i in person_pos:
        person_state = states[person_pos]

        if person_state = 'Ue':
            ....
        elif person_state = 'Un':
            ....
        elif person_state = 'Ae':
            ....
        elif person_state = 'An':
            ....
        elif person_state = 'C':
            ....
        else:
            raise ValueError('Non existing person state')

    # step 5: If the chaos duration is over, uninjured pedestrians regard fallen pedestrians as obstacles and conduct a normal evacuation. There will be no more new fallen pedestrians.

    if T > Tc:
        # C becomes obstacle (= 500)
        # change evacuation strategy of others to normal
        # change to fall = 0

    # step 6: When multiple pedestrians choose to move to the same target, one pedestrian is randomly selected with equal probability to move to the target cell, and other pedestrians remain in their original cells.
        # make list of steps with 1 current pos en 2 next pos. If multiple person have same next pos than equal chance to stay or move

    # step 7: Determine whether the evacuation is over. If there are still uninjured pedestrians in the room, return to step 2 and repeat the simulation process until all the uninjured pedestrians evacuate from the room.

    if not(uninjured in board):
        complete = True