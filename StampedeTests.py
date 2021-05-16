import unittest
from StampedeRaw import Pedestrian, calc_tumble, init_board

# parameters
board_size = 30 # size of board 
num_people = 100 # number of people

exit_locs = ([int(board_size/2-2), int(board_size/2-1), int(board_size/2), int(board_size/2+1)], [0, 0, 0, 0]) # exit locations (middle of left wall)
obstacle_locs = () # No obstacles for now

S_wall = 500
S_exit = 1
mu = 1.5

Ts = 1 # occurrence time of the stampede
Tc = Ts + 50 # chaos duration

kc = 0.5 # sensitivity parameter for tumble factor 
ka = 1 # sensitivity parameter for tumble factor 
ks = 5 # sensitivity parameter for the static field
kd = 1 # sensitivity parameter for the dynamic field
diffusion_factor = 0.3 # aka alpha
decay_factor = 0.3 # aka delta
sight_radius = 5 # perception radius for each person

Srange = 8 # stampede range

state_dic = {'C':1,'Ue':-3,'Un':-4,'Ae':-5,'An':-6}
# each time step == 0.3s

board, person_list = init_board(board_size,num_people,exit_locs,sight_radius,state_dic)

class TumbleTests(unittest.TestCase):

    def test_no_fallen_ped(self):
        person = person_list[0]
        person.location = [13, 15]
        person.direction = [0, 1]
        board[8:19, 10:21] = 0
        board[13, 15] = -6
        new_loc = [coord + 1 for coord in person.location]

        result = calc_tumble(person,sight_radius,ka,kc,board,new_loc)
        self.assertEqual(result, 1) # There is no fallen pedestrian where you want to go so output should be 1 (go ahead and go there)

    def test_fallen_ped_low_dens(self):
        person = person_list[0]
        person.location = [13, 15]
        person.direction = [0, 1]
        board[8:19, 10:21] = 0
        board[13, 15] = -6
        new_loc = [coord + 1 for coord in person.location]
        board[new_loc] = 1

        result = calc_tumble(person,sight_radius,ka,kc,board,new_loc)
        self.assertEqual(result, 0) # There is a fallen pedestrian but low density of people around, so no chance on tripping
    
    def test_fallen_ped_high_dens(self):
        person = person_list[0]
        person.location = [13, 15]
        person.direction = [0, 1]
        board[8:19, 10:21] = -6 # People all around!
        board[13, 15] = -6
        new_loc = [coord + 1 for coord in person.location]
        board[new_loc] = 1

        result = calc_tumble(person,sight_radius,ka,kc,board,new_loc)
        self.assertTrue(result > 0 and result < 1) # There is a fallen pedestrian and high density of people around, so high chance of tripping!

unittest.main()