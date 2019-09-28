import matplotlib.pyplot as pyplot
import transform
import model
import numpy
import sys

def plotV(inV):
    elems = [[' ' for x in range(transform.ARENA_VOL)] for y in
            range(transform.ARENA_VOL)]
    for p in range(transform.ARENA_VOL):
        x,y = transform.pos2_decode(p)
        if inV[model.IN_V_EXIST_FILTER][p] > 0:
            elems[x][y] = "F"
        if inV[model.IN_V_EXIST_ENCRYPTOR][p] > 0:
            elems[x][y] = "E"
        if inV[model.IN_V_EXIST_DESTRUCTOR][p] > 0:
            elems[x][y] = "D"
    grid = ""
    for y in reversed(range(transform.ARENA_SIZE)):
        grid += "["
        for x in range(transform.ARENA_SIZE):
            grid += elems[x][y] + ' '
        grid += "]\n"
    print(grid)

def plotV1(axis, data):
    """
    Plot a encoded half arena density map for the deployment of defensive
    units.
    """
    result = numpy.zeros((transform.ARENA_SIZE, transform.HALF_ARENA))
    for i in range(transform.HALF_ARENA_VOL):
        x, y = transform.pos2_decode(i)
        result[x][y] = data[i]
    axis.imshow(result.T, origin='lower', vmin=0,vmax=2)

def plotV2(axis, data):
    """
    Plot a encoded half arena density map for deploying attack units.
    """
    result = numpy.zeros((transform.ARENA_SIZE, transform.HALF_ARENA))
    for i in range(transform.ARENA_SIZE):
        x, y = transform.pos2_edge_decode(i)
        result[x][y] = data[i]
    axis.imshow(result.T, origin='lower')

def plotOutput(v1, v2):
    fig, axis = pyplot.subplots(nrows=2,ncols=4)
    plotV1(axis[0,0], v1[model.OUT_V1_DELETE])
    axis[0,0].set_title('Deletion')
    plotV1(axis[0,1], v1[model.OUT_V1_PLACE_FILTER])
    axis[0,1].set_title('Filter')
    plotV1(axis[0,2], v1[model.OUT_V1_PLACE_ENCRYPTOR])
    axis[0,2].set_title('Encryptor')
    plotV1(axis[0,3], v1[model.OUT_V1_PLACE_DESTRUCTOR])
    axis[0,3].set_title('Destructor')
    plotV2(axis[1,0], v2[model.OUT_V2_PLACE_PING])
    axis[1,0].set_title('Ping')
    plotV2(axis[1,1], v2[model.OUT_V2_PLACE_EMP])
    axis[1,1].set_title('EMP')
    plotV2(axis[1,2], v2[model.OUT_V2_PLACE_SCRAMBLER])
    axis[1,2].set_title('Scrambler')
    axis[1,3].axis('off')

def plotInitialConfiguration(m = model.create()):
    i1 = numpy.zeros((model.IN_V_SIZE, transform.ARENA_VOL))
    i2 = numpy.zeros(model.IN_S_SIZE)
    i2[model.IN_S_TURN]  = 1 / model.NORMAL_S_TURN
    i2[model.IN_S_SELF_HEALTH]  = 30 / model.NORMAL_S_HEALTH
    i2[model.IN_S_ENEMY_HEALTH] = 30 / model.NORMAL_S_HEALTH
    i2[model.IN_S_SELF_BITS]    =  5 / model.NORMAL_S_BITS
    i2[model.IN_S_ENEMY_BITS]   =  5 / model.NORMAL_S_BITS
    i2[model.IN_S_SELF_CORES]   = 10 / model.NORMAL_S_CORES
    i2[model.IN_S_ENEMY_CORES]  = 10 / model.NORMAL_S_CORES
    result = m.predict([numpy.array([i1,]), numpy.array([i2,])])

    plotOutput(result[0][0], result[1][0])



if __name__ == '__main__':
    if len(sys.argv) <= 1:
        print("Usage: %s test|model" % sys.argv[0])
    elif sys.argv[1] == 'test':
        field = numpy.random.rand(transform.HALF_ARENA_VOL)
        field[transform.pos2_encode((3,10))] = 2
        plotV1(field)
        pyplot.show()
    elif sys.argv[1] == 'model':
        m = model.create()
        if len(sys.argv) >= 3:
            m.load_weights(sys.argv[2])
        plotInitialConfiguration(m)
        pyplot.show()
    elif sys.argv[1] == 'replay':
        inV, inS, outV1, outV2, wV1,wV2= model.loadSample(sys.argv[2])
        if inV.shape[0] == 0:
            print("Unable to load from {}".format(sys.argv[2]))
            exit(0)
        turns = int(sys.argv[3])
        denormal = numpy.array([
            model.NORMAL_S_TURN,
            model.NORMAL_S_HEALTH,
            model.NORMAL_S_BITS,
            model.NORMAL_S_CORES,
            model.NORMAL_S_HEALTH,
            model.NORMAL_S_BITS,
            model.NORMAL_S_CORES,
            ])


        print("Resources: {}".format(inS[turns] * denormal))
        plotV(inV[turns])
        plotOutput(outV1[turns], outV2[turns])
        pyplot.show()
    else:
        print("Unknown mode %s" % sys.argv[1])
