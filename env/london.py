import sys

sys.path.append("..")
import itertools

import numpy as np
from collections import deque

from utils import oneHot_encoding_state


class TowersOfLondon:
    """Implementation of Towers of London game for RL algorithms"""

    def __init__(
        self,
        N,
        max_steps,
        init_state_idx=0,
        init_state=((0, 1), (0, 0), (1, 0)),
        goal_state=((0, 0), (0, 2), (0, 1)),
    ):  # RBG initial typical configuration
        self.discs = N  # n. of balls to be used: 3 for our experiments
        self.n_pegs = N  # number of pegs

        # ===== Generate the state space =======
        # State representation: tuple of (x, y) for each ball
        # Each ball is assigned a unique (x, y) position, where x is peg index and y is position on the peg
        

        self.oneH_s_size = (
            self.discs * self.n_pegs * 2
        )  # needed to determine input to networks
        ## =====================================

        self.goal = goal_state # initialise goal in tuple representation: goal_state is introduced manually at the moment.
        self.init_state = init_state  # initialise starting position

        ## Moves is a tuple where the first entry denotes the peg we are going from and the second entry denotes the peg we are going to
        ## e.g. [0,1] is moving a disk from the first peg (0) to the second peg (1)
        ## here we are denoting all possible moves, checking later whether the move is allowd
        self.moves = list(
            itertools.permutations(list(range(self.n_pegs)), 2)
        )  # No matter the number of disks, there are always n_pegs! (legal/illegal) moves

        self.max_steps = max_steps
        self.reset_check = False
        self.step_counter = 0

    def step(self, action):

        assert self.reset_check, "Need to reset env before taking a step"

        ## Perform a step in tower of London, by passing an action indexing one of the 6 available moves
        ## If an illegal move is chosen, give rwd=-1 and remain in current state
        move = self.moves[action]
        illegal_move = not self._move_allowed(move)  # return False if move is allowed

        self.step_counter += 1
        if not illegal_move:
            moved_state = self._get_moved_state(move)
            if (
                moved_state != self.goal
            ):  ## return rwd=0 if the goal has not been reached (with a legal move)
                rwd = 0
                done = False
                self.c_state = moved_state
            else:  ## return rwd=100 if goal has been reached
                rwd = 100
                done = True
                self.reset_check = False
                self.step_counter = 0
        else:
            ## if selected illegal move, don't terminate state but state in the same state and rwd=-1
            rwd = -100 / 1000
            moved_state = self.c_state
            done = False

        # if reach max step terminate
        if self.step_counter == self.max_steps:
            done = True
            self.reset_check = False
            self.step_counter = 0

        # Compute one-hot representation for new_state to be returned
        oneH_moved_state = oneHot_encoding_state(moved_state, n_integers=self.n_pegs)
        return oneH_moved_state, rwd, done, illegal_move

    def reset(self):
        """Reset always starting from the initial state pre-specified by init_state_idx
        Returns:
            the initial state in one-hot encoding representation
        """
        self.reset_check = True
        self.c_state = self.init_state
         # reset to some initial state, e.g., first state (0,0,0,...), all disks on first peg
        self.oneH_c_state = oneHot_encoding_state(self.c_state, n_integers=self.n_pegs)
        return self.oneH_c_state, self.c_state

    def random_reset(self):
        """Reset starting from a random (legal) inital state
        Returns:
            the initial state in one-hot encoding representation
        """
        self.reset_check = True
        # Make sure don't restart at goal state, need loop since goal state not always = last state in terms of indxes
        while True:
            self.c_state = ()  # temporary initialisation
            availability = self.n_pegs-np.arange(self.n_pegs)
            for disc in range(self.discs):
                peg_index = np.random.choice(np.arange(self.n_pegs)[availability>0])
                self.c_state += ((peg_index, self.n_pegs - availability[peg_index] - peg_index),)
                availability[peg_index]-=1
            if self.c_state != self.goal:
                break
        self.oneH_c_state = oneHot_encoding_state(self.c_state, n_integers=self.n_pegs)
        return self.oneH_c_state, self.c_state

    def current_state(self):
        """Return current state in a list with original representation - i.e., not one-hot"""
        return list(self.c_state)

    def _discs_on_peg(self, peg):
        ## Allows to create a list contatining all the disks that are on that specific peg at the moment (i.e. self.state)
        return [
            (disc,self.c_state[disc][1]) for disc in range(self.discs) if self.c_state[disc][0] == peg
        ]  # add to list only disks that are on that specific peg along with their y coord (disc,disc_y_coord)
    

    def _move_allowed(self, move):
        discs_from = self._discs_on_peg(
            move[0]
        )  # Check what disks are on the peg we want to move FROM
        discs_to = self._discs_on_peg(
            move[1]
        )  # Check what disks are on the peg we want to move TO

        if (
            discs_from
        ):  # Check the list is not empty (i.e. there is at list a disk on the peg we want to move from)
            ## NOTE: Here needs the extra if ... else ... because if disc_to is empty, min() returns an error
            return (
                (len(discs_to) < self.n_pegs - move[1]) if discs_to else True
              )  # return True if we are allowed to make the move (i.e. number of disks on target peg is less than peg capacity (n_pegs - target peg index))
        else:
            return False  # else return False, the move is not allowed

    def _get_moved_state(self, move):
        if self._move_allowed(move):
            disc_to_move = max(self._discs_on_peg(move[0]), key=lambda x: x[1])[0]#pick the disk in highest position on peg we want to move FROM
            moved_state = list(self.c_state)  # take current state
            ## NOTE: since each state dim is a disk (not a peg) then a move only changes that one dim of the state referring to the moved disk
            moved_state[disc_to_move] = (
                move[1],(max(self._discs_on_peg(move[1]), key=lambda x: x[1])[1]+1 if self._discs_on_peg(move[1]) else 0)
                                     )
         #update current state by simply changing the value of the (one) disk that got moved
         #move disc at the lowest possible position on the target peg.
        return tuple(moved_state)  # Return new (moved) state

    def solve_optimal_steps(self, initial_state):
        """
        Returns the minimal number of steps to reach goal_state from initial_state in the given TowersOfLondon env.
        Uses BFS for optimality.
        """
        goal_state = self.goal
        visited = set()
        queue = deque()
        queue.append((initial_state, 0))
        visited.add(initial_state)

        while queue:
            state, steps = queue.popleft()
            if state == goal_state:
                return steps
            # Try all possible moves
            for action, move in enumerate(self.moves):
                # Set self.c_state to current state for move checking
                self.c_state = state
                if self._move_allowed(move):
                    next_state = self._get_moved_state(move)
                    if next_state not in visited:
                        visited.add(next_state)
                        queue.append((next_state, steps + 1))
        return None  # No solution found
    

# Example usage:
# env = TowersOfLondon(N=3, max_steps=100)
# steps = solve_optimal_steps(env.init_state, env.goal, env)
# print("Optimal steps:", steps)


