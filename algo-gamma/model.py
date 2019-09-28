import array, math
import transform
import gamelib

class Policy:

    def __init__(self):
        """
        Represents policy used to evaluate a path.
        """

        self.w_dist_self  = 0.0  # Distance traveled in friendly territory
        self.w_dist_enemy = 1   # Distance traveled in enemy territory
        self.w_shield     = 0.3  # Shielding (Encryptor)
        self.w_damage     = -1   # Damage received from enemy
        self.w_pos_x      = 0   # Favour of right (+) vs left (-)
        self.w_pos_y      = 0   # Favour of edge (+) vs center (-)
        self.w_harass     = 25

        self.damage_bias  = 1.1
        # How many stability points per health point?
        self.ratio_health_stab = 100 


class Path:
    """
    The Path class represents the path a attacking unit can take. After being
    evaluated by the model, the path will have two fields:

    1. damage: Distance * Destructor output
    2. shield: Shielding provided by encryptors

    """
    def __init__(self, px, py, player):
        self.evaluated = False
        self.player = player
        assert len(px) == len(py)
        self.px = px
        self.py = py

        l = len(px)

        # Integrands of the path integral.
        self.damage_dp = array.array('f', [0] * l)
        self.shield_dp = array.array('f', [0] * l)
        # In units of damage * distance
        self.damage = 0
        self.shield = 0
        # Distance spent in the friendly and enemy territories
        self.dist_self = 0
        self.dist_enemy = 0
        self.feasibility = float("-inf")
        self.harass = 0.0
        self.analysed_attack = False

        self.hazard_dp = array.array('f', [0] * l)
        self.hazard = 0
        self.pressure_dp = array.array('f', [0] * l)
        self.analysed_defense = False

        if player == 0:
            self.deadend = (self.py[-1] < (transform.HALF_ARENA - 1))
        else:
            assert player == 1
            self.deadend = (self.py[-1] >= (transform.HALF_ARENA + 1))

    def __len__(self):
        return len(self.px)

    def __eq__(self, other):
        if type(other) != Path: return False
        if self.px != other.px: return False
        if self.py != other.py: return False
        if self.player != other.player: return False
        return True

    def __iter__(self):
        a = [(self.px[i], self.py[i]) for i in range(len(self))].__iter__()
        return a

    def __getitem__(self, i):
        return self.px[i], self.py[i]

    def proximityTest(self, pos, radius):
        """
        Return index in this path for which the path first comes in within
        radius "radius + 0.51" of the given position.

        If returns -1, no such point exists.
        """
        x, y = pos
        radiusSq = (radius + 0.51) ** 2
        for i in range(len(self)):
            rhoSq = (self.px[i] - x) ** 2 + (self.py[i] - y) ** 2
            if rhoSq <= radiusSq:
                return i
        return -1

    def integrate(self, field):
        """
        Expensive function! If you need complex integration schemes consider
        doing this yourself. Integrates on a field, which is a function taking
        a position 0 <= p < ARENA_VOL
        """
        result = 0
        for pos in self:
            p = transform.pos2_encode(pos)
            result += field(p)
        return result


    @property
    def isEdgeReaching(self):
        """
        True if this path ends in an enemy edge.
        """
        return transform.pos2_edge_isOpposing(self[0], self[-1])
    @property
    def isDeadend(self):
        """
        True if path ends in the formation of its deploying player.
        """
        if transform.is_lowerHalf(self.py[0]):
            return transform.is_lowerHalf(self.py[-1] + 1)
        else:
            return transform.is_upperHalf(self.py[-1] - 1)


    def toBytes(self):
        body = (len(self)).to_bytes(4, byteorder='big')
        body += (self.player).to_bytes(1, byteorder='big')
        body += transform.array_to_string(self.px)
        body += transform.array_to_string(self.py)
        body += transform.array_to_string(self.damage_dp)
        body += transform.array_to_string(self.shield_dp)
        body += transform.array_to_string(self.hazard_dp)

        a = array.array('f', [self.feasibility])
        body += transform.array_to_string(a)
        return body

    @classmethod
    def fromBytes(cls, s):
        l = int.from_bytes(s[0:4], 'big'); s = s[4:]
        if l == 0:
            return None, s # A zero corresponds to none.
        player = int.from_bytes(s[0:1], 'big'); s = s[1:]
        px, s = transform.array_from_string(s, 'i')
        py, s = transform.array_from_string(s, 'i')

        result = cls(px = px, py = py, player=player)
        result.damage_dp, s = transform.array_from_string(s, 'f')
        result.shield_dp, s = transform.array_from_string(s, 'f')
        result.hazard_dp, s = transform.array_from_string(s, 'f')
        a, s = transform.array_from_string(s, 'f')
        result.feasibility = a[0]
        result.evaluated = True #(result.feasibility != float('-inf'))

        return result, s

    @classmethod
    def fromGamePath(cls, path):
        px, py = zip(*path) # Transpose
        px = array.array('i', px)
        py = array.array('i', py)
        player = (0 if transform.is_lowerHalf(py[0]) else 1)
        return cls(px = px, py = py, player = player)

    @classmethod
    def group_toBytes(cls, li: list):
        """
        Convert a list of paths to bytes. The list may include null.
        """
        body = (len(li)).to_bytes(4, byteorder='big')
        for p in li:
            if p:
                assert type(p) == Path
                body += p.toBytes()
            else:
                body += (0).to_bytes(4, byteorder='big')
        return body

    @classmethod
    def group_fromBytes(clas, s):
        n = int.from_bytes(s[0:4], 'big'); s = s[4:]
        result = []
        for i in range(n):
            p, s = Path.fromBytes(s)
            result.append(p)
        return result, s


UNIT_TYPE_TO_INDEX = {}

class Model:
    """
    Usage:

    1. readGameState
       
       Gets a game_state object and reads in all of the fields.

    2. readPaths
     
       Determine the order 1 parameters of each path.

    3. analyseAttack

       Determine the order 2 parameters of each path, including
       feasibility, pressure

    4. analyseDefense

       Determine the order 3 parameters of each path including hazard.
    """

    def __init__(self):
        # Health of firewall, encryptor, destructor
        self.stability_F = array.array('f', [0] * transform.ARENA_VOL)
        self.stability_E = array.array('f', [0] * transform.ARENA_VOL)
        self.stability_D = array.array('f', [0] * transform.ARENA_VOL)

        self.number_E_self = 0
        self.number_E_enemy = 0
        self.number_D_self = 0
        self.number_D_enemy = 0
        #
        # Pressure field represents the expected total damage when
        # 1. The enemy randomly samples a path
        # 2. The enemy deploys all bits in the form of EMP's
        #
        self.pressure_self  = array.array('f', [0] * transform.ARENA_VOL)
        self.pressure_enemy = array.array('f', [0] * transform.ARENA_VOL)
        # Attack ranges of decryptors
        self.barrage_self  = array.array('f', [0] * transform.ARENA_VOL)
        self.barrage_enemy = array.array('f', [0] * transform.ARENA_VOL)

        # Stores distance from current position to nearest unit, up to 4.
        # +inf indicates dist > 4
        self.proximity_self  = array.array('f',
                [float(0)] * transform.ARENA_VOL)
        self.proximity_enemy = array.array('f',
                [float(0)] * transform.ARENA_VOL)

        self.li_encryptors_self = []
        self.li_encryptors_enemy = []

        self.flag_pathOutdated = False
        self.resetPaths()

        # Primary attack trajectories
        self.primal_self = None
        self.primal_enemy = None
        # Set of prohibited locations as determined by the primary trajectory.
        self.prohibited_loc = set()

        # Persistent data
        self.policy_self = Policy()
        self.policy_enemy = Policy()

        self.bits_self = 0
        self.bits_enemy = 0
        self.cores_self = 0
        self.cores_enemy = 0

    def resetPaths(self):
        if not self.flag_pathOutdated:
            self.path1_self = []
            self.path1_enemy = []
            # If this is true, the paths are calculated using a historic version
            # of stability fields.
            self.flag_pathOutdated = True

    def evaluatePathThroughput(self, path, unittype: int, n):
        """
        If we send n units down the given path, what is the remaining number
        of units by the type the units reach the end. Assume that the units
        have no attack capabilities

        If the base health of a unit is h and the number of units is m, the
        remaining total HP when the units reach the target is

            (h + shield) * m - damage / speed

        The speed and health are given, so the number of units required is
        
            m = damage / speed / (h+shield)

        Note that we generally don't mix units since they travel at different
        speeds.
        """
        assert path.analysed_attack
        speed = self.SPEED[unittype]
        assert speed > 0

        stab = self.STABILITY[unittype] + path.shield
        
        return n - (self.policy_self.damage_bias * path.damage / speed / stab) \
                - (self.policy_self.w_dist_enemy * path.dist_enemy / speed / stab)




    
    ### State Evaluation Functions ###

    def _evaluatePath(self, path, nEMP: int = 1):
        """
        Populate the shield,damage field of the path. Populate pressure field.
        """
        if not path:
            # Path starts from a occupied unit.
            return float('-inf')

        path.evaluated = True
        player = 0 if transform.is_lowerHalf(path.py[0]) else 1

        path.feasibility = 0.0
        path.shield = 0.0
        path.damage = 0.0
        path.harass = 0.0

        if path.isDeadend:
            path.feasibility = float('-inf')
            return path.feasibility

        # Get list of encryptors
        if player == 0: encs = self.li_encryptors_self.copy()
        else:           encs = self.li_encryptors_enemy.copy()

        EMP_STABILITY = self.STABILITY[UNIT_TYPE_TO_INDEX[EMP]]
        EMP_SPEED     = self.SPEED[UNIT_TYPE_TO_INDEX[EMP]]
        EMP_RANGE     = self.RANGE[UNIT_TYPE_TO_INDEX[EMP]]

        n = len(path)
        for i in range(n):
            x = path.px[i]
            y = path.py[i]
            p = transform.pos2_encode((x,y))

            # Filter the encs list
            ne = len(encs)
            encs = [e for e in encs if \
                transform.distance2(e, (x,y)) >= \
                    self.RANGE[UNIT_TYPE_TO_INDEX[ENCRYPTOR]] + 0.51]
            shield = (ne - len(encs)) * self.ENCRYPTOR_SHIELD
            path.shield_dp[i] = shield
            path.shield += shield

            # Enemy destructor
            if player == 0: damage = self.barrage_enemy[p]
            else:           damage = self.barrage_self[p]
            path.damage_dp[i] = damage
            path.damage += damage

            if transform.is_lowerHalf(y) == (player == 0):
                path.dist_self += 1
            else:
                path.dist_enemy += 1

            nEMP += shield / EMP_STABILITY
            nEMP -= damage / EMP_STABILITY / EMP_SPEED

            path.pressure_dp[i] = nEMP * self.DAMAGE_F_EMP
            if player == 0:
                if self.proximity_enemy[p] == EMP_RANGE:
                    path.harass += 1
                    #gamelib.debug_write('Harassed (Self)')

                # This is always 0
                if transform.is_upperHalf(y):
                    path.pressure_dp[i] -= \
                        self.stability_F[p] + \
                        self.stability_E[p] + \
                        self.stability_D[p]
            else:
                if self.proximity_self[p] == EMP_RANGE:
                    path.harass += 1
                    #gamelib.debug_write('Harassed (Enemy)')

                # This is always 0
                if transform.is_lowerHalf(y):
                    path.pressure_dp[i] -= \
                        self.stability_F[p] + \
                        self.stability_E[p] + \
                        self.stability_D[p]

            AMBIENT_PRESSURE = self.DAMAGE_F_EMP
            path.pressure_dp[i] = max(path.pressure_dp[i], AMBIENT_PRESSURE)

        x_normal, y_normal = path.px[0], path.py[0]
        if player == 0:
            policy = self.policy_self
        else:
            policy = self.policy_enemy
            x_normal, y_normal = transform.pos2_flip((x_normal, y_normal))
        x_normal /= (transform.ARENA_SIZE - 1)
        y_normal /= (transform.HALF_ARENA - 1)
        path.feasibility = policy.w_dist_self  * path.dist_self \
                         - policy.w_dist_enemy * path.dist_enemy \
                         + policy.w_shield     * path.shield \
                         + policy.w_damage     * path.damage \
                         + policy.w_pos_x      * x_normal \
                         + policy.w_pos_y      * y_normal \
                         + policy.w_harass     * path.harass
        return path.feasibility

    def analyseAttack(self):
        # Find optimal path for self and enemy
        p_self = None
        f_self = float('-inf')

        def extract_feas(p):
            if p == None: return float('-inf')
            else:         return p.feasibility
        def softmax(l: list):
            SOFTMAX_COMPRESS = 10
            l = [x / SOFTMAX_COMPRESS for x in l]
            m = max(l)
            if m == float('-inf'):
                return [0] * len(l)
            def condexp(x):
                if x == float('-inf'):
                    return 0
                else:
                    return math.exp(x - m)
            l = [condexp(x - m) for x in l]
            m2 = sum(l)
            l = [x / m2 for x in l]
            return l


        # Max # EMP spawnable
        maxemp_self = self.getNUnitsAffordable(EMP, player=0)
        for path in self.path1_self:
            if not path: continue

            f = self._evaluatePath(path, maxemp_self)
            path.analysed_attack = True
            if f > f_self:
                #gamelib.debug_write("New max: {} at {}".format(f, path.px[0]))
                f_self = f
                p_self = path

        # Use weighted feasibility to get pressure field for the other player
        feas = [extract_feas(p) for p in self.path1_self]
        feas = softmax(feas)
        if max(feas) <= 0.0:
            gamelib.debug_write("Self path maximum feasibility = 0")
        for (f, path) in zip(feas, self.path1_self):
            if f <= 0: continue
            for i in range(len(path)):
                # Calculate contribution
                x,y = path[i]
                p = transform.pos2_encode((x,y))
                self.pressure_enemy[p] += path.pressure_dp[i] * f

        p_enemy = None
        f_enemy = float('-inf')
        maxemp_enemy = self.getNUnitsAffordable(EMP, player=1)
        for path in self.path1_enemy:
            if not path: continue

            f = self._evaluatePath(path, maxemp_self)
            path.analysed_attack = True
            if f > f_enemy:
                f_enemy = f
                p_enemy = path
        # Use weighted feasibility to get pressure field for the other player
        feas = [extract_feas(p) for p in self.path1_enemy]
        feas = softmax(feas)
        if max(feas) <= 0.0:
            gamelib.debug_write("Enemy path maximum feasibility = 0")
        for (f, path) in zip(feas, self.path1_enemy):
            if f <= 0: continue
            for i in range(len(path)):
                # Calculate contribution
                x,y = path[i]
                p = transform.pos2_encode((x,y))
                self.pressure_self[p] += path.pressure_dp[i] * f

        self.primal_self = p_self
        self.primal_enemy = p_enemy

        if not self.primal_self:
            gamelib.debug_write('[Model] No primary trajectory is found!')
            return
        else:
            gamelib.debug_write('[Model] Primal Path start at [{}, {}]'.format(
                self.primal_self.px[0], self.primal_self.py[0]))

    def markTrajectory(self, path):
        if not path:
            return

        for (x,y) in path:
            if transform.is_lowerHalf(y):
                self.prohibited_loc.add((x,y))
    def clearTrajectory(self):
        self.prohibited_loc = set()

    def analyseReactive(self):
        for path in self.path1_self:
            if not path: continue

            path.hazard = 0
            for i in range(len(path)):
                x2, y2 = path[i]
                p = transform.pos2_encode((x2, y2))

                stability = 0
                for (dx, dy) in [(0,1),(0,-1),(1,0),(-1,0)]:
                    stability += self.stability_F[p] \
                            + self.stability_E[p] \
                            + self.stability_D[p] * 2
                path.hazard_dp[i] = self.pressure_enemy[p] - stability
                path.hazard += path.hazard_dp[i]

    def scrambler_protection(self):
        """
        Return ((path, number), (path, number)) for scramblers
        """
        # number of scramblers in each path
        bias_cores = self.cores_enemy / self.COST[UNIT_TYPE_TO_INDEX[ENCRYPTOR]] / 10 \
                + self.number_E_enemy
        bias_cores /= 20
        bias_bits = self.bits_enemy
        nScrambler = int(min(max(self.bits_enemy * (1 + bias_cores) / 20, 1), 2))

        path_collection = self.path1_self

        def allowpath(path):
            if path:
                return path.dist_self < 10
            else:
                return True
        def path_index(path):
            if not path:
                return 0
            else:
                assert path.hazard >= 0
                return max(path.hazard, 0.5)

        paths_left = path_collection[0:transform.HALF_ARENA]
        paths_left = [x for x in paths_left if allowpath(x)]
        paths_left = sorted(paths_left, key=path_index)

        paths_right = path_collection[transform.HALF_ARENA:]
        paths_right = [x for x in paths_right if allowpath(x)]
        paths_right = sorted(paths_right, key=path_index)

        #gamelib.debug_write("Left paths: {}, Right paths: {}".format(len(paths_left), len(paths_right)))

        def getOptimal(col):
            if len(col) == 0:
                return None
            path0 = None
            initial = col[-1]
            if initial:
                for pos in initial:
                    if pos[1] >= min(transform.HALF_ARENA, initial[-1][1]):
                        top_end = pos
                total_path = col
                top_min_dis=100
                for path in total_path:
                    if not path: continue
                    
                    end_dis = path.proximityTest(top_end, 2)
                    if end_dis <= top_min_dis:
                        top_min_dis=end_dis
                        path0 = path
            return path0

        path_l = getOptimal(paths_left)
        path_r = getOptimal(paths_right)

        if not path_l:
            path_l = path_r
            path_r = None

        #second_end_path=None
        #path_2 = paths_left[0]
        #if second_path:
        #    for pos in second_path:
        #        if pos[1]>=min(transform.HALF_ARENA,top_path[-1][1]):
        #            second_end=pos
        #    second_min_dis=100
        #    for path in total_path:
        #        if not path: continue
        #        end_dis=path.proximityTest(second_end,2)
        #        if end_dis<=second_min_dis:
        #            second_min_dis=end_dis
        #            second_end_path=path

        if path_l:
            assert path_l.dist_self < 10
            if path_r:
                assert path_r.dist_self < 10
            if path_r:
                return (path_l, nScrambler), (path_r, nScrambler)
            else:
                return (path_l, nScrambler * 2), (None, 0)
        else:
            return (None, 0), (None, 0)



    def getNUnitsAffordable(self, ty, player: int):
        """
        Returns a fuzzy number, i.e. can return 2.5 units.
        """
        i = UNIT_TYPE_TO_INDEX[ty]
        if i <= 2:
            # Defense
            bits = self.bits_self if player == 0 else self.bits_enemy
            return bits / self.COST[i]
        else:
            # Attack
            cores = self.cores_self if player == 0 else self.cores_enemy
            return cores / self.COST[i]

    def addUnit(self, game_state, pos, unittype):
        """
        Called by spawnDefensiveUnit only. Can only add points in friendly
        territory
        """
        uid = UNIT_TYPE_TO_INDEX[unittype]
        assert transform.is_lowerHalf(pos[1])
        p = transform.pos2_encode(pos)
        stability = self.STABILITY[uid]
        if   uid == UNIT_TYPE_TO_INDEX[FILTER]: 
            self._add_stationary_unit(pos, attacking=False)
            self.stability_F[p] = self.STABILITY[uid]
        elif uid == UNIT_TYPE_TO_INDEX[ENCRYPTOR]:
            self._add_stationary_unit(pos, attacking=False)
            self.stability_E[p] = self.STABILITY[uid]
        elif uid == UNIT_TYPE_TO_INDEX[DESTRUCTOR]:
            self._add_stationary_unit(pos, attacking=True)
            self._add_destructor_contribution(pos)
            self.stability_D[p] = self.STABILITY[uid]



    ### Game State Input ###



    def loadConfig(self, config):
        global FILTER, ENCRYPTOR, DESTRUCTOR, PING, EMP, SCRAMBLER
        FILTER     = config["unitInformation"][0]["shorthand"]
        ENCRYPTOR  = config["unitInformation"][1]["shorthand"]
        DESTRUCTOR = config["unitInformation"][2]["shorthand"]
        PING       = config["unitInformation"][3]["shorthand"]
        EMP        = config["unitInformation"][4]["shorthand"]
        SCRAMBLER  = config["unitInformation"][5]["shorthand"]

        UNIT_TYPE_TO_INDEX[FILTER]     = 0
        UNIT_TYPE_TO_INDEX[ENCRYPTOR]  = 1
        UNIT_TYPE_TO_INDEX[DESTRUCTOR] = 2
        UNIT_TYPE_TO_INDEX[PING]       = 3
        UNIT_TYPE_TO_INDEX[EMP]        = 4
        UNIT_TYPE_TO_INDEX[SCRAMBLER]  = 5

        self.COST      = [0] * 6
        self.RANGE     = [0] * 6
        self.STABILITY = [0] * 6
        self.SPEED     = [0] * 6

        self.DAMAGE_F_EMP = config["unitInformation"][ \
                UNIT_TYPE_TO_INDEX[EMP]]["damageF"]

        for idx in [FILTER,ENCRYPTOR,DESTRUCTOR,PING,EMP,SCRAMBLER]:
            i = UNIT_TYPE_TO_INDEX[idx]
            self.COST[i]      = config["unitInformation"][i]["cost"]
            self.STABILITY[i] = config["unitInformation"][i]["stability"]
            self.RANGE[i]     = config["unitInformation"][i]["range"]
        for idx in [PING,EMP,SCRAMBLER]:
            i = UNIT_TYPE_TO_INDEX[idx]
            self.SPEED[i] = config["unitInformation"][i]["speed"]


        self.ENCRYPTOR_SHIELD = config["unitInformation"][ \
                UNIT_TYPE_TO_INDEX[ENCRYPTOR]]["shieldAmount"]

        # Stores tuples (x, y, r) of points from origin with distance r.
        self.MAX_STATIONARY_RANGE = self.RANGE[UNIT_TYPE_TO_INDEX[EMP]]
        self.CIRCLE = transform.pos2_circle(self.MAX_STATIONARY_RANGE)
        self.CIRCLE_DESTRUCTOR = transform.pos2_circle(\
                self.RANGE[UNIT_TYPE_TO_INDEX[DESTRUCTOR]])


    def readGameState(self, game_state):
        self.bits_self = game_state.get_resource(game_state.BITS, 0)
        self.bits_enemy = game_state.get_resource(game_state.BITS, 1)
        self.cores_self = game_state.get_resource(game_state.CORES, 0)
        self.cores_enemy = game_state.get_resource(game_state.CORES, 1)

        self.number_E_self = 0
        self.number_E_enemy = 0
        self.number_D_self = 0
        self.number_D_enemy = 0
        # Primary attack trajectories
        self.primal_self = None
        self.primal_enemy = None
        # Set of prohibited locations as determined by the primary trajectory.
        self.prohibited_loc = set()
        # Create units map
        for p in range(transform.ARENA_VOL):
            self.stability_F[p] = 0
            self.stability_E[p] = 0
            self.stability_D[p] = 0
            self.pressure_self[p] = 0
            self.pressure_enemy[p] = 0
            self.barrage_self[p] = 0
            self.barrage_enemy[p] = 0
            self.proximity_self[p] = float('+inf')
            self.proximity_enemy[p] = float('+inf')

            x,y = transform.pos2_decode(p)

            units = game_state.game_map[x, y]
            for unit in units:
                stability = unit.stability
                if not unit.stationary: continue

                
                if unit.unit_type == FILTER:
                    self._add_stationary_unit([x,y], attacking=False)
                    self.stability_F[p] = stability
                elif unit.unit_type == ENCRYPTOR:
                    if transform.is_lowerHalf(y):
                        self.number_E_self += 1
                    else:
                        self.number_E_enemy += 1
                    self._add_stationary_unit([x,y], attacking=False)
                    self.stability_E[p] = stability
                elif unit.unit_type == DESTRUCTOR:
                    self._add_stationary_unit([x,y], attacking=True)
                    self._add_destructor_contribution((x,y))
                    self.stability_D[p] = stability


    def readPaths(self, game_state):

        self.resetPaths()

        # Create the pressure map.
        for p in range(transform.ARENA_SIZE):
            # Player 1 perspective
            x,y = transform.pos2_edge_decode(p)
            if not game_state.contains_stationary_unit([x,y]):
                # Determine the target edge from x,y.
                if transform.is_lowerHalf(p):
                    edge = game_state.game_map.TOP_RIGHT
                else:
                    assert transform.is_upperHalf(p)
                    edge = game_state.game_map.TOP_LEFT

                path = game_state.find_path_to_edge([x,y], edge)
                self.path1_self.append(Path.fromGamePath(path))
            else:
                self.path1_self.append(None)

            # Player 2 persepctive
            x,y = transform.pos2_flip((x,y))
            if not game_state.contains_stationary_unit([x,y]):
                # Reminder: When p is lower half on the top edge, it
                # corresponds to TOP_RIGHT!
                if transform.is_lowerHalf(p):
                    edge = game_state.game_map.BOTTOM_LEFT
                else:
                    assert transform.is_upperHalf(p)
                    edge = game_state.game_map.BOTTOM_RIGHT
                path = game_state.find_path_to_edge([x,y], edge)
                self.path1_enemy.append(Path.fromGamePath(path))
            else:
                self.path1_enemy.append(None)

        # Check invariants
        assert len(self.path1_self) == transform.ARENA_SIZE
        assert len(self.path1_enemy) == transform.ARENA_SIZE
        self.flag_pathOutdated = False

    def _add_stationary_unit(self, pos, attacking=True):
        """
        attacking: True if this unit is a destructor
        """
        x, y = pos
        if transform.is_lowerHalf(y):
            for (i, j, r) in self.CIRCLE:
                if not transform.pos2_inbound((x+i,y+j)): continue
                p = transform.pos2_encode((x+i,y+j))

                if r < self.proximity_self[p]:
                    if attacking:
                        self.proximity_self[p] = r
                    else:
                        self.proximity_self[p] = self.MAX_STATIONARY_RANGE
        else:
            for (i, j, r) in self.CIRCLE:
                if not transform.pos2_inbound((x+i,y+j)): continue
                p = transform.pos2_encode((x+i,y+j))

                if r < self.proximity_enemy[p]:
                    if attacking:
                        self.proximity_enemy[p] = r
                    else:
                        self.proximity_enemy[p] = self.MAX_STATIONARY_RANGE

    def _add_destructor_contribution(self, pos):
        (x, y) = pos
        if transform.is_lowerHalf(pos[1]):
            self.number_D_self += 1
            for (dx, dy, r) in self.CIRCLE_DESTRUCTOR:
                if not transform.pos2_inbound((x+dx,y+dy)): continue
                p = transform.pos2_encode((x+dx,y+dy))
                self.barrage_self[p] += 1
        else:
            self.number_D_enemy += 1
            for (dx, dy, r) in self.CIRCLE_DESTRUCTOR:
                if not transform.pos2_inbound((x+dx,y+dy)): continue
                p = transform.pos2_encode((x+dx,y+dy))
                self.barrage_enemy[p] += 1

    def ping_chase_emp(self):
        path_collection = self.path1_self
        max_fea_path = self.primal_self

        if not max_fea_path:
            return [(None, 0), (None, 0)]

        ping_path = max_fea_path
        ping_loc =ping_path[0]
        ping_dist = ping_path.dist_self
        target_loc = ping_path[-ping_dist]
        emp_dist = ping_dist*self.SPEED[UNIT_TYPE_TO_INDEX[EMP]]//self.SPEED[UNIT_TYPE_TO_INDEX[PING]]
        emp_loc = None
        for path in path_collection:
            if not path and not emp_loc:
                emp_loc=ping_loc
            else:
                if target_loc in path:
                    if path[emp_dist-1] == target_loc:
                        emp_loc=path[0]
        given_res = self.bits_self
        if given_res-2*self.COST[UNIT_TYPE_TO_INDEX[EMP]]>0:
            num_of_emp=2
            num_of_ping=int((given_res-2*self.COST[UNIT_TYPE_TO_INDEX[EMP]])//self.COST[UNIT_TYPE_TO_INDEX[PING]])
        else:
            num_of_emp=0
            num_of_ping=0
            
        return [(ping_loc,num_of_ping),(emp_loc,num_of_emp)]

if __name__ == '__main__':
    def test_equal(a, b):
        if a != b:
                print("Test failed: {} != {}".format(a,b))
    def test_nequal(a, b):
        if a == b:
                print("Test failed: {} == {}".format(a,b))

    # Path IO

    px = array.array('i', [1,2,3])
    py = array.array('i', [7,5,6])
    path1 = Path(px, py, player=1)
    path2 = Path(px, py, player=1)
    s = path1.toBytes()

    path1_out, s = Path.fromBytes(s)
    test_equal(path1_out, path2)
    test_equal(len(s), 0)

    px = array.array('i', [1,2,3])
    py = array.array('i', [7,5,6])
    path1 = Path(px, py, player=0)
    path2 = Path(px, py, player=1)
    s = path1.toBytes()

    path1_out, s = Path.fromBytes(s)
    test_nequal(path1_out, path2)
    test_equal(len(s), 0)

    # Path group IO
    px = array.array('i', [1,2,3])
    py = array.array('i', [7,5,6])
    path1 = Path(px, py, player=1)
    px = array.array('i', [7,8])
    py = array.array('i', [7,9])
    path2 = Path(px, py, player=0)

    g = [path1, None, path2]
    s = Path.group_toBytes(g)
    g_out, s = Path.group_fromBytes(s)
    test_equal(g, g_out)

    print("=== Ignore the following errors ===")
    test_equal("Test1", "Test2")
    print("=== Assertion System test complete ===")

    print("Test Complete. Exit.")



