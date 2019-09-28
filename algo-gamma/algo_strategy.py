import gamelib
import random
import math
import warnings
from sys import maxsize
import sys
import array
import transform
import model
import json

if True:
    FILE_STABILITY_F     = "sampo.stability_f.array"
    FILE_STABILITY_E     = "sampo.stability_e.array"
    FILE_STABILITY_D     = "sampo.stability_d.array"
    FILE_PRESSURE_SELF   = "sampo.pressure_self.array"
    FILE_PRESSURE_ENEMY  = "sampo.pressure_enemy.array"
    FILE_BARRAGE_SELF    = "sampo.barrage_self.array"
    FILE_BARRAGE_ENEMY   = "sampo.barrage_enemy.array"
    FILE_PROXIMITY_SELF  = "sampo.proximity_self.array"
    FILE_PROXIMITY_ENEMY = "sampo.proximity_enemy.array"

    FILE_PATH1_SELF      = "sampo.path1_self.array"
    FILE_PATH1_ENEMY     = "sampo.path1_enemy.array"
    
    FILE_ERROR = "sampo.stderr.log"
else:
    FILE_STABILITY_F      = None
    FILE_STABILITY_E      = None
    FILE_STABILITY_D      = None
    FILE_PRESSURE_SELF    = None
    FILE_PRESSURE_ENEMY   = None
    FILE_BARRAGE_SELF     = None
    FILE_BARRAGE_ENEMY    = None
    FILE_PROXIMITY_SELF   = None
    FILE_PROXIMITY_ENEMY  = None
    FILE_PATH1_SELF       = None
    FILE_PATH1_ENEMY      = None
    FILE_ERROR            = None


UNIT_TYPE_TO_INDEX = {}

class AlgoStrategy(gamelib.AlgoCore):
    def __init__(self):
        super().__init__()
        random.seed()
        
        self.m = model.Model()

        if FILE_STABILITY_F:
            self.file_stability_F = open(FILE_STABILITY_F, 'wb')
        if FILE_STABILITY_E:
            self.file_stability_E = open(FILE_STABILITY_E, 'wb')
        if FILE_STABILITY_D:
            self.file_stability_D = open(FILE_STABILITY_D, 'wb')
        if FILE_PRESSURE_SELF:
            self.file_pressure_self = open(FILE_PRESSURE_SELF, 'wb')
        if FILE_PRESSURE_ENEMY:
            self.file_pressure_enemy = open(FILE_PRESSURE_ENEMY, 'wb')
        if FILE_BARRAGE_SELF:
            self.file_barrage_self = open(FILE_BARRAGE_SELF, 'wb')
        if FILE_BARRAGE_ENEMY:
            self.file_barrage_enemy = open(FILE_BARRAGE_ENEMY, 'wb')
        if FILE_PROXIMITY_SELF:
            self.file_proximity_self = open(FILE_PROXIMITY_SELF, 'wb')
        if FILE_PROXIMITY_ENEMY:
            self.file_proximity_enemy = open(FILE_PROXIMITY_ENEMY, 'wb')
        if FILE_PATH1_SELF:
            self.file_path1_self = open(FILE_PATH1_SELF, 'wb')
        if FILE_PATH1_ENEMY:
            self.file_path1_enemy = open(FILE_PATH1_ENEMY, 'wb')



    def on_game_start(self, config):
        self.config = config
        global FILTER, ENCRYPTOR, DESTRUCTOR, PING, EMP, SCRAMBLER
        FILTER = config["unitInformation"][0]["shorthand"]
        ENCRYPTOR = config["unitInformation"][1]["shorthand"]
        DESTRUCTOR = config["unitInformation"][2]["shorthand"]
        PING = config["unitInformation"][3]["shorthand"]
        EMP = config["unitInformation"][4]["shorthand"]
        SCRAMBLER = config["unitInformation"][5]["shorthand"]

        UNIT_TYPE_TO_INDEX[FILTER] = 0
        UNIT_TYPE_TO_INDEX[ENCRYPTOR] = 1
        UNIT_TYPE_TO_INDEX[DESTRUCTOR] = 2
        UNIT_TYPE_TO_INDEX[PING] = 3
        UNIT_TYPE_TO_INDEX[EMP] = 4
        UNIT_TYPE_TO_INDEX[SCRAMBLER] = 5

        self.m.loadConfig(config)
        self.defense_init()

    def defense_init(self):
        # initialize defense needed variable, should be in on_game_start

        # variable for defense_start
        self.defense_start_list = [
            # corner protection
            [FILTER, [
                [0, 13], [27, 13], [1, 13], [26, 13], [2, 13], [25, 13]
                ]],
            # basic destructor
            [DESTRUCTOR, [
                [4, 12], [23, 12], [10, 10], [17, 10]
                ]],
            # protect destructor
            [FILTER, [
                [4, 13], [23, 13], [10, 11], [17, 11], [5, 12], [22, 12] 
                ]],
            # filter wall
            [FILTER, [
                [3, 12], [24, 12], [4, 11], [23, 11], [5, 10], [22, 10],
                [6, 9], [21, 9], [7, 9], [20, 9], [8, 9], [19, 9], 
                [9, 9], [18, 9], [10, 9], [17, 9], [11, 9], [16, 9], 
                [12, 9], [15, 9], [13, 9], [14, 9]
                ]]
            ]

        # record any non-filter unit (remove or destuctor)
        self.defense_basic_dict = {} 

    def on_turn(self, turn_state):
        """
        This function is called every turn with the game state wrapper as
        an argument. The wrapper stores the state of the arena and has methods
        for querying its state, allocating your current resources as planned
        unit deployments, and transmitting your intended deployments to the
        game engine.
        """
        game_state = gamelib.GameState(self.config, turn_state)
        gamelib.debug_write('Gamma version 1.1, turn {}'.format(game_state.turn_number))

        if self.defense_start_list != []:
            self.defense_start(game_state)
        
        self.defense_basic(game_state)

#
#        
#        gamelib.debug_write('Sampo turn {}'.format(game_state.turn_number))
#        #game_state.suppress_warnings(True)  #Uncomment this line to suppress warnings.
#
#        if True: #game_state.turn_number == 0:
#            self.deployDefenseInitial(game_state)
#
#        # Turn cycle:
#        # 1. Analyse game state
#        # 2. Obtain most feasible path for self & enemy
#        # 3. Service self path & thwart enemy path
#        # 4. Re-analyse game state
#        # 5. Deploy defensive units
#        self.m.readGameState(game_state)
#
#        self.m.readPaths(game_state)
#        self.m.analyseAttack()
#        max_hazard_path = self.m.analyseReactive()
#
#        assert not self.m.flag_pathOutdated
#        # Must dump all statistics now. They get reset afterwards.
#        if FILE_STABILITY_F:
#            self.file_stability_F.write(transform.array_to_string(self.m.stability_F))
#        if FILE_STABILITY_E:
#            self.file_stability_E.write(transform.array_to_string(self.m.stability_E))
#        if FILE_STABILITY_D:
#            self.file_stability_D.write(transform.array_to_string(self.m.stability_D))
#        if FILE_PRESSURE_SELF:
#            self.file_pressure_self.write(transform.array_to_string(self.m.pressure_self))
#        if FILE_PRESSURE_ENEMY:
#            self.file_pressure_enemy.write(transform.array_to_string(self.m.pressure_enemy))
#        if FILE_BARRAGE_SELF:
#            self.file_barrage_self.write(transform.array_to_string(self.m.barrage_self))
#        if FILE_BARRAGE_ENEMY:
#            self.file_barrage_enemy.write(transform.array_to_string(self.m.barrage_enemy))
#        if FILE_PROXIMITY_SELF:
#            self.file_proximity_self.write(transform.array_to_string(self.m.proximity_self))
#        if FILE_PROXIMITY_ENEMY:
#            self.file_proximity_enemy.write(transform.array_to_string(self.m.proximity_enemy))
#
#        if FILE_PATH1_SELF:
#            self.file_path1_self.write(model.Path.group_toBytes(self.m.path1_self))
#        if FILE_PATH1_ENEMY:
#            self.file_path1_enemy.write(model.Path.group_toBytes(self.m.path1_enemy))
#
#
#        # Cache the trajectory so it does not get invalidated.
#        trajectory = self.m.primal_self
#
#        self.servicePath(game_state, trajectory)
#        self.deployDefence(game_state)
#
#        if trajectory:
#            self.deployAttack(game_state, trajectory)
#        if max_hazard_path:
#            x0,y0 = max_hazard_path[0]
#            game_state.attempt_spawn(SCRAMBLER, [x0, y0], 1)
#            game_state.attempt_spawn(SCRAMBLER, [x0, y0], 1)
#
#        #self.m.readPaths(game_state)
#
#

        game_state.submit_turn()

    def defense_start(self, game_state):
        gamelib.debug_write('defense_start')
        for i in range(0, len(self.defense_start_list)):
            level = self.defense_start_list[i]
            unit_type = level[0]
            for j in range(0, len(level[1])):
                position = level[1][j]
                if game_state.attempt_spawn(unit_type, position) == 0:
                    # no resource available, remove any spawned unit
                    level[1] = level[1][j:]
                    self.defense_start_list = self.defense_start_list[i:]
                    gamelib.debug_write(json.dumps(self.defense_start_list))
                    return
        # all unit successfully spawned, set defense_start_list to be empty
        self.defense_start_list = []

    def defense_basic(self, game_state):
        game_map = game_state.game_map

        def add_destructor(x, y):
            if game_map[x, 13] != []:
            # location is not empty
                return
            
            if (x, y) in self.defense_basic_dict:
                if "R" in self.defense_basic_dict[(x, y)]:
                    # we intentionly remove it, don't do anything
                    return
                if not "D" in self.defense_basic_dict[(x, y)]:
                    # add desctructor info to dict
                    self.defense_basic_dict[(x, y)].append("D")
            game_state.attempt_spawn(DESTRUCTOR, [x, y])

        for i in {0, 27, 1, 26, 2, 25}:
            add_destructor(i, 13)
        
        for i in range(4):
            add_destructor(3 + i, 12 - i)
            add_destructor(24 - i, 12 - i)
        
        for i in range(7):
            add_destructor(7 + i, 9)
            add_destructor(20 - i, 9)

    def defense_encryptor(self, game_state):
        num = 0
        for i in range(0, 4):
            num += game_state.attempt_spawn(ENCRYPTOR, [7 + i, 8 - i])
            if (num > 2):
                return
            num += game_state.attempt_spawn(ENCRYPTOR, [20 - i, 8 - i])
            if (num > 2):
                return

    def defense_enhance(self, game_state):
        pass

    def servicePath(self, game_state, path):
        """
        Build encryptors to service the primary path
        """
        if not path:
            gamelib.debug_write('No servicable primal path exists!')
            return

        gamelib.debug_write('Servicing Primal.')
        n = len(path)
        for i in range(n):
            if game_state.get_resource(game_state.CORES) < self.m.COST[UNIT_TYPE_TO_INDEX[ENCRYPTOR]]:
                # No more cores
                return
            x = path.px[i]
            y = path.py[i]
            self.spawnDefensiveUnit(game_state, ENCRYPTOR, [x+1, y+1])
            self.spawnDefensiveUnit(game_state, ENCRYPTOR, [x-1, y+1])

            if transform.is_lowerHalf(x):
                self.spawnDefensiveUnit(game_state, ENCRYPTOR, [x+1, y])
            else:
                self.spawnDefensiveUnit(game_state, ENCRYPTOR, [x-1, y])
            self.spawnDefensiveUnit(game_state, ENCRYPTOR, [x, y+1])

    def deployDefence(self, game_state):
        """
        Deploy defensive units to thwart enemy path based on the pressure.
        """
        pass

    def deployAttack(self, game_state, trajectory):
        """
        Launch ping-emp rush along the primary path
        """
        assert trajectory

        x0 = trajectory.px[0]
        y0 = trajectory.py[0]
        gamelib.debug_write('Deploying Attack from [{},{}]'.format(x0,y0))
        bits = game_state.get_resource(game_state.BITS)

        # Evaluate throughput using pings and emp.
        nPings = int(bits // self.m.COST[UNIT_TYPE_TO_INDEX[PING]])
        nEMPs  = int(bits // self.m.COST[UNIT_TYPE_TO_INDEX[EMP]])

        nPings_survive = self.m.evaluatePathThroughput(trajectory,
            UNIT_TYPE_TO_INDEX[PING], nPings)
        nEMP_survive = self.m.evaluatePathThroughput(trajectory,
            UNIT_TYPE_TO_INDEX[PING], nEMPs)

        if nEMP_survive >= 2:
            game_state.attempt_spawn(EMP, [x0, y0], nEMPs)
        elif nPings_survive >= 4:
            game_state.attempt_spawn(PING, [x0, y0], nPings)


    def deployDefenseInitial(self, game_state):
        # LogicFan's Tortoise
        SCHEMA = [
           {'type': FILTER, 'location': [0, 13]},
           {'type': FILTER, 'location': [1, 13]},
           {'type': FILTER, 'location': [2, 13]},
           {'type': FILTER, 'location': [3, 13]},
           {'type': FILTER, 'location': [27, 13]},
           {'type': FILTER, 'location': [26, 13]},
           {'type': FILTER, 'location': [25, 13]},
           {'type': FILTER, 'location': [24, 13]},
           {'type': FILTER, 'location': [4, 12]},
           {'type': FILTER, 'location': [23, 12]},
           # Destructor
           {'type': DESTRUCTOR, 'location': [4, 11]},
           {'type': DESTRUCTOR, 'location': [10, 11]},
           {'type': DESTRUCTOR, 'location': [17, 11]},
           {'type': DESTRUCTOR, 'location': [23, 11]},
           # Protect for destructor
           {'type': FILTER, 'location': [5, 11]}, # First destructor
           {'type': FILTER, 'location': [22, 11]},

           {'type': FILTER, 'location': [9, 11]},
           {'type': FILTER, 'location': [10, 12]},
           {'type': FILTER, 'location': [11, 11]}, # Second
           {'type': FILTER, 'location': [16, 11]},
           {'type': FILTER, 'location': [17, 12]},
           {'type': FILTER, 'location': [18, 11]}, # Third
        ]
        for unit in SCHEMA:
            self.spawnDefensiveUnit(game_state, unit['type'], unit['location'])

    def on_action_frame(self, turn_string):
        """
        Analyse breaches and correct the path
        """

    def spawnDefensiveUnit(self, game_state, ty, location):
        """
        Must use this function to spawn defensive unit since it corrects the
        model.
        """
        assert type(location) == list
        x,y = location
        if not game_state.game_map.in_arena_bounds([x,y]) \
                or transform.is_upperHalf(y):
            return
        if (x,y) in self.m.prohibited_loc:
            gamelib.debug_write( \
                    '[spawnDefensiveUnit] Attempt to spawn on trajectory: {}.'.format(location))
            return
        if game_state.contains_stationary_unit(location):
            gamelib.debug_write( \
                    '[spawnDefensiveUnit] Defensive unit exists: {} at {}.'.format(ty, location))
            return

        result = game_state.attempt_spawn(ty, location)
        if result == 0:
            gamelib.debug_write( \
                    '[spawnDefensiveUnit] Could not spwan at location {}'.format(location))

        self.m.resetPaths()

if __name__ == "__main__":
    # DO NOT DIVERT stdout
    if FILE_ERROR:
        sys.stderr = open(FILE_ERROR, 'w')


    algo = AlgoStrategy()
    algo.start()
