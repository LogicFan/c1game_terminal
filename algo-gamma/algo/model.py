import sys
import tensorflow as tf
import tensorflow.keras as keras
import numpy
import math
import transform

# Input indices

# Input V
# (#Entry) * ARENA_VOL vector
IN_V_HEALTH_FILTER = 0
IN_V_HEALTH_ENCRYPTOR = 1
IN_V_HEALTH_DESTRUCTOR = 2
IN_V_EXIST_FILTER = 3
IN_V_EXIST_ENCRYPTOR = 4
IN_V_EXIST_DESTRUCTOR = 5
IN_V_SIZE = 6

# Input S
#
# Bits & Cores are normalised by /5. Turn number by /10
IN_S_TURN = 0
IN_S_SELF_HEALTH = 1
IN_S_SELF_BITS = 2
IN_S_SELF_CORES = 3
IN_S_ENEMY_HEALTH = 4
IN_S_ENEMY_BITS = 5
IN_S_ENEMY_CORES = 6
IN_S_SIZE = 7

# Output 1: Action set 1
OUT_V1_DELETE = 0
OUT_V1_PLACE_FILTER = 1
OUT_V1_PLACE_ENCRYPTOR = 2
OUT_V1_PLACE_DESTRUCTOR = 3
OUT_V1_SIZE = 4

# Output 2: Action set 2
OUT_V2_PLACE_PING = 0
OUT_V2_PLACE_EMP = 1
OUT_V2_PLACE_SCRAMBLER = 2
OUT_V2_SIZE = 3

NORMAL_S_TURN = 10
NORMAL_S_BITS = 5
NORMAL_S_CORES = 5
NORMAL_S_HEALTH = 30




def predictOn(model, inV, inS):
    result = model.predict([numpy.array([inV,]), numpy.array([inS,])])
    return result[0][0], result[1][0]

def preproc_density(game_state, densityV1, densityV2):
    # Process board
    for i in range(transform.HALF_ARENA_VOL):
        x,y = transform.pos2_decode(i)
        if game_state.contains_stationary_unit((x,y)):
            densityV1[OUT_V1_PLACE_FILTER][i] = 0
            densityV1[OUT_V1_PLACE_ENCRYPTOR][i] = 0
            densityV1[OUT_V1_PLACE_DESTRUCTOR][i] = 0
        else:
            densityV1[OUT_V1_DELETE][i] = 0
    # Process edge strip
    for i in range(transform.ARENA_SIZE):
        [x,y] = transform.pos2_edge_decode(i)
        if game_state.contains_stationary_unit((x,y)):
            densityV2[OUT_V2_PLACE_PING][i] = 0
            densityV2[OUT_V2_PLACE_EMP][i] = 0
            densityV2[OUT_V2_PLACE_SCRAMBLER][i] = 0
    numpy.clip(densityV1, a_min=0, a_max=None, out=densityV1)
    numpy.clip(densityV2, a_min=0, a_max=None, out=densityV2)
    return densityV1, densityV2

def probabilistic_round(x: float):
    """
    Probabilistically round a floating point number. e.g.
    4.5 -> (4, 5) with 50% chance each.
    3.1 -> (3, 4) with 10%, and 90%
    """
    frac, i = math.modf(x)
    i = int(i)
    return i + numpy.random.binomial(1, frac)

def reverse_V(inV):
    return numpy.flip(numpy.copy(inV), -1)
def reverse_S(inS):
    inS2 = numpy.copy(inS)
    inS2[IN_S_SELF_HEALTH]  = inS[IN_S_ENEMY_HEALTH]
    inS2[IN_S_SELF_BITS]    = inS[IN_S_ENEMY_BITS]
    inS2[IN_S_SELF_CORES]   = inS[IN_S_ENEMY_CORES]
    inS2[IN_S_ENEMY_HEALTH] = inS[IN_S_SELF_HEALTH]
    inS2[IN_S_ENEMY_BITS]   = inS[IN_S_SELF_BITS]
    inS2[IN_S_ENEMY_CORES]  = inS[IN_S_SELF_CORES]
    return inS2

def totalScore_V1(inS):

    def depreciation(x):
        return min(-x+2, 0)
    score = 0
    score += inS[IN_S_SELF_HEALTH] * 10
    score += depreciation(inS[IN_S_SELF_CORES])
    #score -= inS[IN_S_ENEMY_BITS]
    #score -= inS[IN_S_ENEMY_CORES]
    return score
def totalScore_V2(inS):
    def depreciation(x):
        # Prevent this algorithm from stockpiling bits/cores
        return math.tanh(x // 3) * 2
    score = 0
    score += inS[IN_S_SELF_HEALTH] * 10
    score += depreciation(inS[IN_S_SELF_BITS])
    score -= inS[IN_S_ENEMY_HEALTH] * 10
    #score -= inS[IN_S_ENEMY_BITS]
    #score -= inS[IN_S_ENEMY_CORES]
    return score

def loadSample(path_base):
    """
    path_base must be devoid of .{A,B}.{in,out},{V1,V2}.npy suffixes.
    """
    path_inV = path_base + ".in.V.npy"
    path_inS = path_base + ".in.S.npy"
    path_A_outV1 = path_base + ".A.out.V1.npy"
    path_A_outV2 = path_base + ".A.out.V2.npy"
    path_B_outV1 = path_base + ".B.out.V1.npy"
    path_B_outV2 = path_base + ".B.out.V2.npy"

    try:
        a_inV = numpy.load(path_inV)[:-1]
        a_inS = numpy.load(path_inS)[:-1]
        a_outV1 = numpy.load(path_A_outV1)[:-1]
        a_outV2 = numpy.load(path_A_outV2)[:-1]
        b_outV1 = numpy.load(path_B_outV1)[:-1]
        b_outV2 = numpy.load(path_B_outV2)[:-1]

        # Determine winner to add bias to weights
        fh1 = a_inS[-1][IN_S_SELF_HEALTH]
        fh2 = a_inS[-1][IN_S_ENEMY_HEALTH]
        diffbias = (fh1 - fh2) * 5
        a_bias = diffbias
        b_bias = -diffbias

        a_weights_V1 = numpy.apply_along_axis(totalScore_V1, -1, a_inS)
        a_weights_V1 = numpy.diff(a_weights_V1) + a_bias
        a_weights_V2 = numpy.apply_along_axis(totalScore_V2, -1, a_inS)
        a_weights_V2 = numpy.diff(a_weights_V2) + a_bias

        # Flip inV along axis to obtain second set of training data.
        b_inV = reverse_V(a_inV)
        b_inS = numpy.apply_along_axis(reverse_S, -1, a_inS)
        b_weights_V1 = numpy.apply_along_axis(totalScore_V1, -1, b_inS)
        b_weights_V1 = numpy.diff(b_weights_V1) + b_bias
        b_weights_V2 = numpy.apply_along_axis(totalScore_V2, -1, b_inS)
        b_weights_V2 = numpy.diff(b_weights_V2) + b_bias

        return (numpy.concatenate([a_inV, b_inV]),
                numpy.concatenate([a_inS[:-1], b_inS[:-1]]),
                numpy.concatenate([a_outV1, b_outV1]),
                numpy.concatenate([a_outV2, b_outV2]),
                numpy.concatenate([a_weights_V1, b_weights_V1]),
                numpy.concatenate([a_weights_V2, b_weights_V2])
                )
    except Exception as e:
        print(e)
        return (numpy.empty((0, IN_V_SIZE, transform.ARENA_VOL)),\
               numpy.empty((0, IN_S_SIZE)), \
               numpy.empty((0, OUT_V1_SIZE, transform.HALF_ARENA_VOL)), \
               numpy.empty((0, OUT_V2_SIZE, transform.ARENA_SIZE)), \
               numpy.empty((0)), \
               numpy.empty((0)))

def printSample(a_inV, a_inS, a_outV1, a_outV2):
    denormal = numpy.array([
        NORMAL_S_TURN,
        NORMAL_S_HEALTH,
        NORMAL_S_BITS,
        NORMAL_S_CORES,
        NORMAL_S_HEALTH,
        NORMAL_S_BITS,
        NORMAL_S_CORES,
        ])
    T_TURN = 3

    if a_inV.shape[0] == 0:
        return

    elems = [[' ' for x in range(transform.ARENA_SIZE)] for y in
            range(transform.ARENA_SIZE)]
    for p in range(transform.ARENA_VOL):
        x,y = transform.pos2_decode(p)
        if a_inV[T_TURN][IN_V_EXIST_FILTER][p] > 0:
            elems[y][x] = "F"
        if a_inV[T_TURN][IN_V_EXIST_ENCRYPTOR][p] > 0:
            elems[y][x] = "E"
        if a_inV[T_TURN][IN_V_EXIST_DESTRUCTOR][p] > 0:
            elems[y][x] = "D"


    drops = [[' ' for x in range(transform.ARENA_SIZE)] for y in
            range(transform.ARENA_SIZE)]
    for p in range(transform.HALF_ARENA_VOL):
        x,y = transform.pos2_decode(p)
        if a_outV1[T_TURN][OUT_V1_PLACE_FILTER][p] > 0:
            drops[y][x] = 'F'
        if a_outV1[T_TURN][OUT_V1_PLACE_ENCRYPTOR][p] > 0:
            drops[y][x] = 'E'
        if a_outV1[T_TURN][OUT_V1_PLACE_DESTRUCTOR][p] > 0:
            drops[y][x] = 'D'
        if a_outV1[T_TURN][OUT_V1_DELETE][p] > 0:
            drops[y][x] = 'x'
    for p in range(transform.ARENA_SIZE):
        x,y = transform.pos2_edge_decode(p)
        if a_outV2[T_TURN][OUT_V2_PLACE_PING][p] > 0:
            drops[y][x] = 'p'
        if a_outV2[T_TURN][OUT_V2_PLACE_EMP][p] > 0:
            drops[y][x] = 'e'
        if a_outV2[T_TURN][OUT_V2_PLACE_SCRAMBLER][p] > 0:
            drops[y][x] = 's'
    grid = ["[" + " ".join(e) + "] [" + " ".join(d) + "]"
            for (e,d) in zip(elems, drops)]
    grid = "\n".join(reversed(grid))
    print("   STAGE" + " " * (transform.ARENA_SIZE*2) + "DROPS")
    print(grid)
    print("S Before: {}".format(a_inS[T_TURN] * denormal))
    print("S After: {}".format(a_inS[T_TURN+1] * denormal))


    

def create():
    ARENA_VOL      = transform.ARENA_VOL
    HALF_ARENA_VOL = transform.HALF_ARENA_VOL
    ARENA_SIZE     = transform.ARENA_SIZE
    REPRESENTATION_SIZE = 6 * ARENA_VOL
    REPRESENTATION_SIZE_2 = 6 * HALF_ARENA_VOL

    inputV = keras.Input(shape=(IN_V_SIZE, ARENA_VOL), name="iV")

    reg = keras.regularizers.l2(0.01)
    procV = keras.layers.Reshape((ARENA_VOL * IN_V_SIZE,))(inputV)
    procV = keras.layers.Dense(ARENA_VOL * IN_V_SIZE, \
            activation="relu", kernel_regularizer=reg)(procV)
    #procV = keras.layers.Dense(1200, activation="relu")(procV) # old
    #procV = keras.layers.Dense(600, activation="relu")(procV) # old

    inputS = keras.Input(shape=(IN_S_SIZE,), name="iS")

    procS = keras.layers.Dense(8, activation="relu")(inputS)
    #procS = keras.layers.Dense(10, activation="relu")(procS) # old

    core = keras.layers.concatenate([procV, procS])
    core = keras.layers.Dense(REPRESENTATION_SIZE, kernel_regularizer=reg)(core)
    core = keras.layers.LeakyReLU(alpha=0.01)(core)
    core = keras.layers.Dense(REPRESENTATION_SIZE, kernel_regularizer=reg)(core)
    core = keras.layers.LeakyReLU(alpha=0.01)(core)
    core = keras.layers.Dense(REPRESENTATION_SIZE, kernel_regularizer=reg)(core)
    core = keras.layers.LeakyReLU(alpha=0.01)(core)
    core = keras.layers.Dense(REPRESENTATION_SIZE_2, kernel_regularizer=reg)(core)
    core = keras.layers.LeakyReLU(alpha=0.01)(core)
    core = keras.layers.Dense(REPRESENTATION_SIZE_2, kernel_regularizer=reg)(core)
    core = keras.layers.LeakyReLU(alpha=0.01)(core)
    core = keras.layers.Dense(REPRESENTATION_SIZE_2, kernel_regularizer=reg)(core)
    core = keras.layers.LeakyReLU(alpha=0.01)(core)

    procO1 = keras.layers.Dense(OUT_V1_SIZE * HALF_ARENA_VOL, # new
             activation="relu", kernel_regularizer=reg)(core)
    procO1 = keras.layers.Dense(OUT_V1_SIZE * HALF_ARENA_VOL,
             activation="sigmoid", kernel_regularizer=reg)(procO1)
    procO1 = keras.layers.Reshape((OUT_V1_SIZE, HALF_ARENA_VOL), name="V1")(procO1)

    procO2 = keras.layers.Dense(OUT_V2_SIZE * ARENA_SIZE, # new
            activation="relu", kernel_regularizer=reg)(core)
    procO2 = keras.layers.Dense(OUT_V2_SIZE * ARENA_SIZE,
            activation="relu", kernel_regularizer=reg)(procO2)
    procO2 = keras.layers.Reshape((OUT_V2_SIZE, ARENA_SIZE), name="V2")(procO2)
    
    model = keras.Model(
            inputs=[inputV, inputS],
            outputs=[procO1, procO2]
    )
    #model._make_predict_function()
    #model._make_test_function()
    #model._make_train_function()
    return model

if __name__ == "__main__":
    if len(sys.argv) <= 1:
        print("Usage: %s test|diag" % sys.argv[0])
    elif sys.argv[1] == 'test':
        model = create()

        i1 = numpy.zeros((IN_V_SIZE, transform.ARENA_VOL))
        i2 = numpy.zeros(IN_S_SIZE)
        i2[IN_S_SELF_HEALTH] = 1
        i2[IN_S_ENEMY_HEALTH] = 1
        i2[IN_S_SELF_BITS] = 1
        i2[IN_S_ENEMY_BITS] = 1
        i2[IN_S_SELF_CORES] = 2
        i2[IN_S_ENEMY_CORES] = 2
        print("Input 1: {}".format(i1.shape))
        print("Input 2: {}".format(i2.shape))

        if len(sys.argv) >= 3:
            model.load_weights(sys.argv[2])
        result = model.predict([numpy.array([i1,]), numpy.array([i2,])])
        assert len(result) == 2
        print("Shape 0: {}".format(result[0][0].shape))
        print("Shape 1: {}".format(result[1][0].shape))
    elif sys.argv[1] == 'diag':
        print(keras.__version__)
        model = create()
        print(model.summary())
        keras.utils.plot_model(model, "model.png", show_shapes=True)
    elif sys.argv[1] == 'sample':
        inV,inS,outV1,outV2,wV1, wV2 = loadSample(sys.argv[2])
        printSample(inV,inS,outV1,outV2)
    else:
        print("Unknown mode: {}".format(sys.argv[1]))
