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
import functools 

if len(sys.argv) >= 2 and sys.argv[1] == 'debug':
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

        self.attack_stage = 0
        self.attack_config = [] # (direction, attack_type, attack_num, support)
        self.last_attack_stage = 0

    def defense_init(self):
        # initialize defense needed variable, should be in on_game_start

        # variable for defense_start
        self.defense_start_list = [
            # corner protection
            [FILTER, [
                [0, 13], [27, 13], [1, 13], [26, 13], [2, 13], [25, 13]
                ]],
            # destructor wall
            [DESTRUCTOR, [
                    [2, 12], [25, 12],
                    [5, 10], [22, 10],
                    [8, 8], [19, 8],
                    [12, 8], [15, 8],
                ]],
            [DESTRUCTOR, [
                # corner
                [1, 12], [26, 12], [3, 12], [24, 12], 
                # inclined
                [4, 11], [23, 11], [7, 8], [20, 8],
                # plane
                [9, 8], [18, 8], [10, 8], [17, 8], 
                [13, 8], [14, 8]
                ]]
            ]

        self.defense_door_list = [[6, 9], [21, 9], [11, 8], [16, 8]]

        # record any non-filter unit (remove or destuctor)
        self.defense_basic_dict = {} 
        self.defense_basic_complete = True

    def on_turn(self, turn_state):
        """
        This function is called every turn with the game state wrapper as
        an argument. The wrapper stores the state of the arena and has methods
        for querying its state, allocating your current resources as planned
        unit deployments, and transmitting your intended deployments to the
        game engine.
        """
        game_state = gamelib.GameState(self.config, turn_state)
        gamelib.debug_write('Gamma version 1.3, turn {}'.format(game_state.turn_number))

        ##########################################################
        if (game_state.my_health <= 8) and (game_state.get_resource(game_state.BITS) >= 15):
            self.last_attack(game_state)
            game_state.submit_turn()
            return

        if self.last_attack_stage == 1:
            self.last_attack(game_state)
            game_state.submit_turn()
            return

        self.m.clearTrajectory()

        ne, np = self.numberEMPPing(game_state)
        reserve = 10 if np > 0 else 0

        if self.defense_start_list != []:
            self.defense_start(game_state)
        else:
            self.defense_conner(game_state)
            self.defense_basic(game_state, reserve)
            # self.defense_corner_dtor(game_state, reserve)
        
        # 
        # if self.defense_basic_complete:
        #     self.defense_encryptor(game_state)
        # 
        # self.defense_enhance(game_state)

        #self.attack_finish(game_state)
        #self.attack_action(game_state)
        #self.attack_prepare(game_state)


        
        gamelib.debug_write('Gamma turn {}'.format(game_state.turn_number))
        #game_state.suppress_warnings(True)  #Uncomment this line to suppress warnings.

        # Turn cycle:
        # 1. Analyse game state
        # 2. Obtain most feasible path for self & enemy
        # 3. Service self path & thwart enemy path
        # 4. Re-analyse game state
        # 5. Deploy defensive units
        self.m.readGameState(game_state)

        self.m.readPaths(game_state)
        self.m.analyseAttack()
        gamelib.debug_write('Calculated pressure for turn {}'.format(game_state.turn_number))
        self.m.analyseReactive()

        assert not self.m.flag_pathOutdated
        # Must dump all statistics now. They get reset afterwards.
        if FILE_STABILITY_F:
            self.file_stability_F.write(transform.array_to_string(self.m.stability_F))
        if FILE_STABILITY_E:
            self.file_stability_E.write(transform.array_to_string(self.m.stability_E))
        if FILE_STABILITY_D:
            self.file_stability_D.write(transform.array_to_string(self.m.stability_D))
        if FILE_PRESSURE_SELF:
            self.file_pressure_self.write(transform.array_to_string(self.m.pressure_self))
        if FILE_PRESSURE_ENEMY:
            self.file_pressure_enemy.write(transform.array_to_string(self.m.pressure_enemy))
        if FILE_BARRAGE_SELF:
            self.file_barrage_self.write(transform.array_to_string(self.m.barrage_self))
        if FILE_BARRAGE_ENEMY:
            self.file_barrage_enemy.write(transform.array_to_string(self.m.barrage_enemy))
        if FILE_PROXIMITY_SELF:
            self.file_proximity_self.write(transform.array_to_string(self.m.proximity_self))
        if FILE_PROXIMITY_ENEMY:
            self.file_proximity_enemy.write(transform.array_to_string(self.m.proximity_enemy))

        if FILE_PATH1_SELF:
            self.file_path1_self.write(model.Path.group_toBytes(self.m.path1_self))
        if FILE_PATH1_ENEMY:
            self.file_path1_enemy.write(model.Path.group_toBytes(self.m.path1_enemy))


        if game_state.turn_number < 3:
            scram1 = None
            scram_n1 = 0
            scram2 = None
            scram_n2 = 0
        else:
            (scram1, scram_n1), (scram2, scram_n2) = self.m.scrambler_protection()

        # Cache the trajectory so it does not get invalidated.
        trajectory = self.m.primal_self

        if trajectory:
            if scram_n1 > 0 or scram_n2 > 0: reserve_bits = 2
            else:                            reserve_bits = 0

            flag = self.deployAttack(game_state, trajectory, reserve=reserve_bits)
            if flag:
                # Attack succeess
                cores_remain = 4 if self.m.bits_enemy > 10 else 0
                self.m.markTrajectory(trajectory)
                self.servicePath(game_state, trajectory, cores_remain=cores_remain)

                # Only 2 scramblers!
                scram_n1 = min(scram_n1, 1)
                scram_n2 = min(scram_n2, 1)
        if scram_n1:
            x,y = scram1[0]
            self.m.markTrajectory(scram1)
            # Mirror attack trajectory
            game_state.attempt_spawn(SCRAMBLER, [x,y], scram_n1)
        if scram_n2:
            x,y = scram2[0]
            self.m.markTrajectory(scram2)
            # Mirror attack trajectory
            game_state.attempt_spawn(SCRAMBLER, [x,y], scram_n2)


        #self.m.readPaths(game_state)
        if game_state.get_resource(game_state.BITS, 1) >= 10:
            self.close_door(game_state)
        game_state.submit_turn()

    def last_attack(self, game_state):
        if self.last_attack_stage == 0:
            for y in range(0, 14):
                for x in range(13 - y, 15 + y):
                    game_state.attempt_remove([x, y])
            self.last_attack_stage = 1
            return

        if self.last_attack_stage == 1:
            for loc in [[15, 2], [14, 2]]:
                game_state.attempt_spawn(ENCRYPTOR, loc)
            for i in range(2, 14):
                game_state.attempt_spawn(ENCRYPTOR, [15 - i, i])
            for i in range(3, 14):
                game_state.attempt_spawn(ENCRYPTOR, [16 - i, i])
            for i in range(0, 11):
                game_state.attempt_spawn(DESTRUCTOR, [27 - i, 13 - i])
            
            emp_num = int((game_state.get_resource(game_state.BITS) * 0.3) // self.m.COST[UNIT_TYPE_TO_INDEX[EMP]])
            ping_num = int((game_state.get_resource(game_state.BITS) * 0.7) // self.m.COST[UNIT_TYPE_TO_INDEX[PING]])
            game_state.attempt_spawn(EMP, [14, 0], emp_num)
            game_state.attempt_spawn(PING, [15, 1], ping_num)
            return

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
                    return
        # all unit successfully spawned, set defense_start_list to be empty
        self.defense_start_list = []
    def defense_conner(self, game_state):
        for i in range(0, 3):
            game_state.attempt_spawn(FILTER, [0 + i, 13])
            game_state.attempt_spawn(FILTER, [27 + i, 13])

    def defense_corner_dtor(self, game_state, reserve):
        corner_dtor_list = [[2, 11], [25, 11], [3, 11], [24, 11]]
        for unit in corner_dtor_list:
            if game_state.get_resource(game_state.CORES) <= reserve:
                return
            game_state.attempt_spawn(DESTRUCTOR, unit)

    def defense_basic(self, game_state, reserve):

        destructor_list = []

        def destructor_health(x, y):
            game_map = game_state.game_map
            if game_map[x, y] == []:
                return 0
            else:
                return game_map[x, y][0].stability

        def compare(x):
            return x[0]

        for i in range(0, 3):
            destructor_list.append(
                (destructor_health(1 + i, 12), 1 + i, 12))
            destructor_list.append(
                (destructor_health(26 - i, 12), 26 - i, 12))
        for i in range(0, 2):
            destructor_list.append(
                (destructor_health(4 + i, 11 - i), 4 + i, 11 - i))
            destructor_list.append(
                (destructor_health(23 - i, 11 - i), 23 - i, 11 - i))
        for i in [0, 1, 2, 3, 5, 6]:
            destructor_list.append(
                (destructor_health(7 + i, 8), 7 + i, 8))
            destructor_list.append(
                (destructor_health(20 - i, 8), 20 - i, 8))
        
        destructor_list.sort(key = compare)
        gamelib.debug_write(json.dumps(destructor_list))

        for unit in destructor_list:
            if unit[0] == 0:
                game_state.attempt_spawn(DESTRUCTOR, [unit[1], unit[2]])
                gamelib.debug_write("D location {}, {}".format(unit[1], unit[2]))
            else:
                break

        for unit in destructor_list:
            if game_state.get_resource(game_state.CORES) <= reserve:
                return
            game_state.attempt_spawn(FILTER, [unit[1], unit[2] + 1])
            gamelib.debug_write("F location {}, {}".format(unit[1], unit[2]))

    def close_door(self, game_state):
        for unit in self.defense_door_list:
            self.spawnDefensiveUnit(game_state, DESTRUCTOR, unit)
            game_state.attempt_remove(unit)
    
    def attack_prepare(self, game_state):
        if self.attack_stage != 0:
            return

        # calculate the enemy defense level
        # def enemy_defense_level(direction):
        #     p1 = [[[0, 14]], 
        #           [[27, 14]]]
        #     p2 = 

        # check conditon, change it later
        if game_state.get_resource(game_state.BITS) > 12:
            #this is a two-element list with tuples indicates the locations of
            #pings and emps
            attack_loc=self.m.ping_chase_emp()
        else:
            return

        
        #game_state.attempt_remove(prepare_list[self.attack_config[0]])
        self.attack_stage = 1

    def attack_action(self, game_state):
        if self.attack_stage != 1:
            return
        
        direction = self.attack_config[0]
        unit_type = self.attack_config[1]
        attack_num = self.attack_config[2]
        supprot_num = self.attack_config[3]

        support_list = [[19, 5], [8, 5]]
        attack_list = [[20, 6], [7, 6]]
        filter_list = [[21, 7], [6, 7]]

        game_state.attempt_spawn(FILTER, filter_list[direction])
        game_state.attempt_remove(filter_list[direction])
        game_state.attempt_spawn(unit_type, support_list[direction], supprot_num)
        game_state.attempt_spawn(unit_type, attack_list[direction], attack_num)
        self.attack_stage = 2

    def attack_finish(self, game_state):
        if self.attack_stage != 2:
            return
        
        edge_list = [[0, 13], [1, 13], [26, 13], [27, 13]]
        game_state.attempt_spawn(FILTER, edge_list)
        self.attack_stage = 0

    def servicePath(self, game_state, path, cores_remain: int):
        """
        Build encryptors to service the primary path
        """
        if not path:
            gamelib.debug_write('No servicable primal path exists!')
            return

        gamelib.debug_write('Servicing Primal.')

        cores = game_state.get_resource(game_state.CORES) - cores_remain
        n = len(path)
        k = int(cores // self.m.COST[UNIT_TYPE_TO_INDEX[ENCRYPTOR]])

        for (x,y) in path:
            for (dx, dy) in [(1,1),(0,1),(-1,1)]:
                if k == 0:
                    return
                if self.spawnDefensiveUnit(game_state, ENCRYPTOR, [x+dx, y+dy]):
                    k -= 1
                    game_state.attempt_remove([[x+dx,y+dy]])
        for (x,y) in path:
            for (dx, dy) in [(1,0),(0,0),(-1,0)]:
                if k == 0:
                    return
                if self.spawnDefensiveUnit(game_state, ENCRYPTOR, [x+dx, y+dy]):
                    k -= 1
                    game_state.attempt_remove([[x+dx,y+dy]])
        for (x,y) in path:
            for (dx, dy) in [(1,-1),(0,-1),(-1,-1)]:
                if k == 0:
                    return
                if self.spawnDefensiveUnit(game_state, ENCRYPTOR, [x+dx, y+dy]):
                    k -= 1
                    game_state.attempt_remove([[x+dx,y+dy]])

    def numberEMPPing(self, game_state, reserve=0):
        bits = game_state.get_resource(game_state.BITS) - reserve
        destruct = min(self.m.number_D_enemy, 10)
        nEMPs = int(2 + destruct * 0.2)
        nPings = int((bits - self.m.COST[UNIT_TYPE_TO_INDEX[EMP]] * nEMPs) \
                // self.m.COST[UNIT_TYPE_TO_INDEX[PING]])

        if nPings >= 5:
            return nEMPs, nPings
        else:
            return 0,0

    def deployAttack(self, game_state, trajectory, reserve=0):
        """
        Launch ping-emp rush along the primary path
        """
        assert trajectory

        x0 = trajectory.px[0]
        y0 = trajectory.py[0]
        gamelib.debug_write('Deploying Attack from [{},{}]'.format(x0,y0))

        # Evaluate throughput using pings and emp.
        nEMPs, nPings = self.numberEMPPing(game_state, reserve=reserve)

        if nEMPs > 0 and nPings > 0:
            game_state.attempt_spawn(EMP, [x0, y0], int(nEMPs))
            game_state.attempt_spawn(PING, [x0, y0], int(nPings))
            return True
        else:
            return False

        #nPings_survive = self.m.evaluatePathThroughput(trajectory,
        #    UNIT_TYPE_TO_INDEX[PING], nPings)
        #nEMP_survive = self.m.evaluatePathThroughput(trajectory,
        #    UNIT_TYPE_TO_INDEX[EMP], nEMPs)

        #gamelib.debug_write('Survival rate: {}'.format(nEMP_survive))



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
        Must use this function to spawn defensive unit since it consults the
        model.
        """
        assert type(location) == list
        x,y = location
        if not game_state.game_map.in_arena_bounds([x,y]) \
                or transform.is_upperHalf(y):
            return False
        if (x,y) in self.m.prohibited_loc:
            gamelib.debug_write( \
                    '[spawnDefensiveUnit] Attempt to spawn on trajectory: {}.'.format(location))
            return False
        if game_state.contains_stationary_unit(location):
            gamelib.debug_write( \
                    '[spawnDefensiveUnit] Defensive unit exists: {} at {}.'.format(ty, location))
            return False

        result = game_state.attempt_spawn(ty, location)
        if result == 0:
            gamelib.debug_write( \
                    '[spawnDefensiveUnit] Could not spwan at location {}'.format(location))
            return False

        #self.m.addUnit(game_state, location, ty)
        self.m.resetPaths()
        return True

if __name__ == "__main__":
    # DO NOT DIVERT stdout
    if FILE_ERROR:
        sys.stderr = open(FILE_ERROR, 'w')


    algo = AlgoStrategy()
    algo.start()
