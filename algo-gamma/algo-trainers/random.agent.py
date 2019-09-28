import gamelib
import random
import math
import warnings
import sys
import transform
import numpy as np

"""
Most of the algo code you write will be in this file unless you create new
modules yourself. Start by modifying the 'on_turn' function.

Advanced strategy tips: 

Additional functions are made available by importing the AdvancedGameState 
class from gamelib/advanced.py as a replacement for the regular GameState class 
in game.py.

You can analyze action frames by modifying algocore.py.

The GameState.map object can be manually manipulated to create hypothetical 
board states. Though, we recommended making a copy of the map to preserve 
the actual current map state.
"""

class AlgoStrategy(gamelib.AlgoCore):
    def __init__(self):
        super().__init__()
        random.seed()

    def on_game_start(self, config):
        """ 
        Read in config and perform any initial setup here 
        """
        gamelib.debug_write('Configuring your custom algo strategy...')
        self.config = config
        self.nFilters = 0
        global FILTER, ENCRYPTOR, DESTRUCTOR, PING, EMP, SCRAMBLER
        FILTER = config["unitInformation"][0]["shorthand"]
        ENCRYPTOR = config["unitInformation"][1]["shorthand"]
        DESTRUCTOR = config["unitInformation"][2]["shorthand"]
        PING = config["unitInformation"][3]["shorthand"]
        EMP = config["unitInformation"][4]["shorthand"]
        SCRAMBLER = config["unitInformation"][5]["shorthand"]


    def on_turn(self, turn_state):
        """
        This function is called every turn with the game state wrapper as
        an argument. The wrapper stores the state of the arena and has methods
        for querying its state, allocating your current resources as planned
        unit deployments, and transmitting your intended deployments to the
        game engine.
        """
        game_state = gamelib.GameState(self.config, turn_state)
        gamelib.debug_write('Random: Performing turn {}'.format(game_state.turn_number))
        #game_state.suppress_warnings(True)  #Uncomment this line to suppress warnings.

        self.buildDefenses(game_state)
        self.deployAttack(game_state)

        game_state.submit_turn()

    def trySpawn(self, game_state, t, loc):
        if game_state.can_spawn(t, loc):
            game_state.attempt_spawn(t, loc)
            return True
        return False

    def buildDefenses(self, game_state):
        turn_nFilters = 0
        def randomLoc():
            n = random.randrange(0, transform.HALF_ARENA_VOL)
            return transform.pos2_decode(n)
        def buildFilter():
            if game_state.number_affordable(FILTER) == 0 or (self.nFilters > 28 and turn_nFilters > 2):
                return False
            loc = randomLoc()
            flag = self.trySpawn(game_state, FILTER, loc)
            if flag:
                ++self.nFilters
            return True
        def buildEncrypt():
            if game_state.number_affordable(ENCRYPTOR) == 0:
                return False
            loc = randomLoc()
            self.trySpawn(game_state, ENCRYPTOR, loc)
            return True
        def buildDestructor():
            if game_state.number_affordable(DESTRUCTOR) == 0:
                return False
            loc = randomLoc()
            self.trySpawn(game_state, DESTRUCTOR, loc)
            return True
        p = [0.333,0.334,0.333]
        funcs = [buildFilter, buildEncrypt, buildDestructor]
        for i in range(np.random.geometric(0.2)):
            if not p:
                break
            assert len(funcs) == len(p)
            choice = np.random.choice(range(len(funcs)), p=p)
            result = funcs[choice]()
            if not result:
                del p[choice]
                del funcs[choice]
                if p:
                    total = sum(p)
                    assert total > 0.0
                    p = [x / total for x in p]

    def deployAttack(self, game_state):
        def randomLoc():
            n = random.randrange(0, transform.ARENA_SIZE)
            return transform.pos2_edge_decode(n)
        def dPing():
            if game_state.number_affordable(PING) < 3:
                return False
            loc = randomLoc()
            if game_state.can_spawn(PING, loc, 3):
                game_state.attempt_spawn(PING, loc, 3)
            return True
        def dEMP():
            if game_state.number_affordable(EMP) == 0:
                return False
            loc = randomLoc()
            if game_state.can_spawn(EMP, loc, 1):
                game_state.attempt_spawn(EMP, loc, 1)
            return True
        def dScrambler():
            if game_state.number_affordable(SCRAMBLER) == 0:
                return False
            loc = randomLoc()
            if game_state.can_spawn(SCRAMBLER, loc, 1):
                game_state.attempt_spawn(SCRAMBLER, loc, 1)
            return True
        p = [0.333,0.334,0.333]
        funcs = [dPing, dEMP, dScrambler]
        for i in range(np.random.geometric(0.2)):
            if not p:
                break
            assert len(funcs) == len(p)
            choice = np.random.choice(range(len(funcs)), p=p)
            result = funcs[choice]()
            if not result:
                del p[choice]
                del funcs[choice]
                if p:
                    total = sum(p)
                    assert total > 0.0
                    p = [x / total for x in p]


if __name__ == "__main__":
    algo = AlgoStrategy()
    algo.start()
