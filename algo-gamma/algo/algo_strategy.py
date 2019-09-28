import gamelib
import math, numpy
import warnings
import sys, os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import model, transform
import itertools
import uuid
import tensorflow

DEBUG_FILE = "dirichlet_out.log"

class AlgoStrategy(gamelib.AlgoCore):
    def __init__(self):
        super().__init__()
        self.m = model.create()
        self.debugOut = open(DEBUG_FILE, "w+")
        #tensorflow.keras.backend.set_floatx('float32')
        try:
            self.m.load_weights(sys.argv[1])
        except:
            pass

    def on_game_start(self, config):
        """ 
        Read in config and perform any initial setup here 
        """
        gamelib.debug_write('Configuring your custom algo strategy...')
        self.config = config
        global FILTER, ENCRYPTOR, DESTRUCTOR, PING, EMP, SCRAMBLER
        FILTER = config["unitInformation"][0]["shorthand"]
        ENCRYPTOR = config["unitInformation"][1]["shorthand"]
        DESTRUCTOR = config["unitInformation"][2]["shorthand"]
        PING = config["unitInformation"][3]["shorthand"]
        EMP = config["unitInformation"][4]["shorthand"]
        SCRAMBLER = config["unitInformation"][5]["shorthand"]


    def on_turn(self, turn_state):
        game_state = gamelib.GameState(self.config, turn_state)
        gamelib.debug_write('Dirichlet Turn {}'.format(game_state.turn_number))
        #game_state.suppress_warnings(True)
        gamelib.debug_write("Turn {}\n".format(game_state.turn_number))
        self.debugWrite("Turn {}\n".format(game_state.turn_number))

        inS = self.game_state_to_input_S(game_state)
        gamelib.debug_write("inS, ");
        inV = self.game_state_to_input_V(game_state)
        gamelib.debug_write("inV, ");

        assert inV.shape == (model.IN_V_SIZE, transform.ARENA_VOL)
        assert inS.shape == (model.IN_S_SIZE,)

        #with self.tf_session.as_default():
        densityV1, densityV2 = model.predictOn(self.m, inV, inS)
        gamelib.debug_write("Model prediction complete\n");
        self.debugWrite("Model prediction complete\n");

        assert densityV1.shape == (model.OUT_V1_SIZE, transform.HALF_ARENA_VOL)
        assert densityV2.shape == (model.OUT_V2_SIZE, transform.ARENA_SIZE)

        densityV1, densityV2 = model.preproc_density(game_state, densityV1, densityV2)
        gamelib.debug_write("Density preprocessing complete\n")

        gamelib.debug_write("Checkpoint 0: {},{}\n".format(densityV1.shape, densityV2.shape))
        quantityV1 = numpy.sum(densityV1, axis=-1)
        quantityV2 = numpy.sum(densityV2, axis=-1)

        gamelib.debug_write("Checkpoint 1: {},{}\n".format(quantityV1.shape, quantityV2.shape))
        assert quantityV1.shape == (model.OUT_V1_SIZE,)
        assert quantityV2.shape == (model.OUT_V2_SIZE,)

        gamelib.debug_write("Checkpoint 1.2\n");
        quantities = numpy.concatenate([quantityV1, quantityV2])
        gamelib.debug_write("Checkpoint 1.4\n");
        quantities = list(numpy.vectorize(model.probabilistic_round)(quantities))
        gamelib.debug_write("Checkpoint 2\n");

        # Normalise all densities
        densityV1 /= densityV1.sum(axis=-1)[..., numpy.newaxis]
        densityV2 /= densityV2.sum(axis=-1)[..., numpy.newaxis]
        gamelib.debug_write("Checkpoint 3\n");

        def removeDefense():
            gamelib.debug_write("Func: RemoveDefense\n")
            k = numpy.random.choice( \
                    range(transform.HALF_ARENA_VOL), \
                    p = densityV1[model.OUT_V1_DELETE])
            [x, y] = transform.pos2_decode(choice)
            if game_state.contains_stationary_unit([x,y]):
                game_state.attempt_remove([[x,y]])
            return True
        def placeDefense(ty, key, text: str):
            def f():
                if game_state.number_affordable(ty) == 0:
                    return False
                # Sample from random location
                k = numpy.random.choice( \
                        range(transform.HALF_ARENA_VOL), \
                        p = densityV1[key])
                x,y = transform.pos2_decode(k)
                if game_state.can_spawn(ty, [x,y]):
                    game_state.attempt_spawn(ty, [[x,y]])
                else:
                    self.log_spawnFailure(text, [x,y])
                return True
            return f
        def placeAttack(ty, key, text: str):
            def f():
                if game_state.number_affordable(ty) == 0:
                    return False
                # Sample from random location
                k = numpy.random.choice( \
                        range(transform.ARENA_SIZE), \
                        p = densityV2[key])
                x,y = transform.pos2_edge_decode(k)
                if game_state.can_spawn(ty, [x,y], 1):
                    game_state.attempt_spawn(ty, [[x,y]], 1)
                else:
                    self.log_spawnFailure(text, [x,y])
                return True
            return f

        gamelib.debug_write("Checkpoint 3.9\n");
        funcs = [
            removeDefense,
            placeDefense(FILTER,    model.OUT_V1_PLACE_FILTER,    "FILTER"),
            placeDefense(ENCRYPTOR, model.OUT_V1_PLACE_ENCRYPTOR, "ENCRYPTOR"),
            placeDefense(DESTRUCTOR, model.OUT_V1_PLACE_DESTRUCTOR, "DESTRUCTOR"),
            placeAttack( PING,      model.OUT_V2_PLACE_PING,      "PING"),
            placeAttack( EMP,       model.OUT_V2_PLACE_EMP,       "EMP"),
            placeAttack( SCRAMBLER, model.OUT_V2_PLACE_SCRAMBLER, "SCRAMBLER"),
        ]
        gamelib.debug_write("Checkpoint 4\n");
        assert len(funcs) == len(quantities)

        # Total number of spawn attempts
        attempts = numpy.sum(quantities)
        gamelib.debug_write("Attempts: {}, v={}\n".format(attempts, quantities))
        self.debugWrite("Attempts: {}, v={}\n".format(attempts, quantities))

        # Filter the functions which have 0 quantity
        mask = [ q > 0 for q in quantities ]
        quantities = list(itertools.compress(quantities, mask))
        quantities = [x for x in quantities]
        funcs = list(itertools.compress(funcs, mask))

        for i in range(int(attempts)):
            # If we have somehow exhausted the quantity, quit
            if len(quantities) == 0:
                break

            # Randomly choose an option and perform it.
            nOptions = len(funcs)
            assert nOptions == len(quantities)
            choice = numpy.random.choice(range(nOptions))
            result = funcs[choice]()
            quantities[choice] -= 1
            if quantities[choice] < 0.99 or not result:
                del quantities[choice]
                del funcs[choice]

        # Stategy here #
        game_state.submit_turn()

    def game_state_to_input_V(self, gameState):
        result = numpy.zeros((model.IN_V_SIZE, transform.ARENA_VOL))
        for i in range(transform.ARENA_VOL):
            x,y = transform.pos2_decode(i)
            for unit in gameState.game_map[x, y]:
                if unit.stationary:
                    health = unit.stability / unit.max_stability
                    if unit.unit_type == FILTER:
                        result[model.IN_V_HEALTH_FILTER][i] = health
                        result[model.IN_V_EXIST_FILTER][i] = 1
                    elif unit.unit_type == ENCRYPTOR:
                        result[model.IN_V_HEALTH_ENCRYPTOR][i] = health
                        result[model.IN_V_EXIST_ENCRYPTOR][i] = 1
                    elif unit.unit_type == DESTRUCTOR:
                        result[model.IN_V_HEALTH_DESTRUCTOR][i] = health
                        result[model.IN_V_EXIST_DESTRUCTOR][i] = 1
                    break
        return result
    
    def game_state_to_input_S(self, gameState):
        result = numpy.ndarray(model.IN_S_SIZE)
    
        result[model.IN_S_TURN] = gameState.turn_number / model.NORMAL_S_TURN
        result[model.IN_S_SELF_HEALTH] = gameState.my_health / model.NORMAL_S_HEALTH
        result[model.IN_S_SELF_BITS] = \
                gameState.get_resource(gameState.BITS, 0) / model.NORMAL_S_BITS
        result[model.IN_S_SELF_CORES] = \
                gameState.get_resource(gameState.CORES, 0) / model.NORMAL_S_CORES
        result[model.IN_S_ENEMY_HEALTH] = gameState.enemy_health / model.NORMAL_S_HEALTH
        result[model.IN_S_ENEMY_BITS] = \
                gameState.get_resource(gameState.BITS, 1) / model.NORMAL_S_BITS
        result[model.IN_S_ENEMY_CORES] = \
                gameState.get_resource(gameState.CORES, 1) / model.NORMAL_S_CORES
        return result
    
    
    def log_spawnFailure(self, t: str, loc):
        pass
    def debugWrite(self, x):
        if DEBUG_FILE:
            self.debugOut.write(x)



if __name__ == "__main__":
    if True:
        sys.stderr = open('/tmp/error.{}.txt'.format(uuid.uuid4()), 'w')
    algo = AlgoStrategy()
    algo.start()
