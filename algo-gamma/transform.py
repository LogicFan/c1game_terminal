import math, sys
import array

ARENA_SIZE = 28
HALF_ARENA = ARENA_SIZE // 2
HALF_ARENA_VOL = 210
ARENA_VOL = HALF_ARENA_VOL * 2

# Operations that transform the game state into a input vector.

def is_lowerHalf(y: int):
    return y < HALF_ARENA
def is_upperHalf(y: int):
    return y >= HALF_ARENA

def pos2_inbound(pos):
    x, y = pos
    return (x + y >= HALF_ARENA - 1) \
       and (x + y <= ARENA_SIZE + HALF_ARENA - 1) \
       and (x - y <= HALF_ARENA) \
       and (y - x <= HALF_ARENA)

def pos2_flip(pos):
    x, y = pos
    assert pos2_inbound(pos)
    return (ARENA_SIZE - 1 - x, ARENA_SIZE - 1 - y)
def pos2_encode(pos):
    """
    Converts a arena position (x,y) into a serialised position n
    """
    assert 0 <= pos[1] and pos[1] < ARENA_SIZE
    def pos2_encode_lower(pos):
        x, y = pos
        assert 0 <= x and x < ARENA_SIZE
        assert 0 <= y and y < HALF_ARENA
        left_x = HALF_ARENA - y - 1
        return (y * (y + 1)) + (x - left_x)
    # Detect if y is on the upper half plane.
    if pos[1] >= HALF_ARENA:
        return ARENA_VOL - pos2_encode_lower(pos2_flip(pos)) - 1
    else:
        return pos2_encode_lower(pos)
def pos2_decode(n):
    """
    Converts a serialised position n into an arena position x,y
    """
    def pos2_decode_lower(n):
        # Round n/2 to nearest triangular number
        #
        # The general formula of converting a triangular number k to its index
        # is:
        #
        #     1
        #     - * (-1 + sqrt(1 + 8k))
        #     2
        #
        # This gives the y coordinate. For x we can do the reverse of
        # pos2_encode_lower
        y = int(0.5 * (math.sqrt(1+4*n) - 1))
        row_begin = y*(y+1)
        left_x = HALF_ARENA - 1 - y
        x = n - row_begin + left_x
        return x,y

    if n >= HALF_ARENA_VOL:
        return pos2_flip(pos2_decode_lower(ARENA_VOL - n - 1))
    else:
        return pos2_decode_lower(n)

def pos2_edge_decode(n):
    """
    The edge decode function are only available for the lower edge
    """
    assert 0 <= n and n < ARENA_SIZE
    if n >= HALF_ARENA:
        return [n, (n - HALF_ARENA)]
    else:
        return [n, (HALF_ARENA - 1 - n)]

def pos2_edge_on(pos):
    """
    Determine which edge is pos on.

    Return
    """
    x, y = pos
    if x + y == HALF_ARENA - 1:
        return 1 # Bot-left
    elif x + y == ARENA_SIZE + HALF_ARENA - 1:
        return 4 # Top-right
    elif x - y == HALF_ARENA:
        return 2 # Bot-right
    elif y - x == HALF_ARENA:
        return 3 # Top-left
    else:
        return 0 # Not on an edge

def pos2_edge_isOpposing(pos1, pos2):
    return (pos2_edge_on(pos1) + pos2_edge_on(pos2)) == 5

def pos2_circle(radius: float):
    """
    A unit with a given range affects all locations who's centers are
    within that range + 0.51 so we add 0.51 here
    """
    radiusSq = (radius + 0.51) ** 2
    result = []
    for i in range(int(-radius), int(radius + 1)):
        for j in range(int(-radius), int(radius + 1)):
            rho = i ** 2 + j ** 2
            if rho <= radiusSq:
                rhoi = int(math.sqrt(rho))
                result.append((i,j,rhoi))
    return result


def distance2(location_1, location_2):
    """Euclidean distance
    (Copied from game_map)

    Args:
        * location_1: An arbitrary location
        * location_2: An arbitrary location

    Returns:
        The euclidean distance between the two locations

    """
    x1, y1 = location_1
    x2, y2 = location_2

    return math.sqrt((x1 - x2)**2 + (y1 - y2)**2)

def array_to_string(a):
    body = a.tobytes()
    n = len(body)
    return (n).to_bytes(4, byteorder='big') + body
def array_from_string(s, ty='i'):
    # Returns:
    # 1. Data
    # 2. Remainder of string
    n = int.from_bytes(s[0:4], 'big')
    a = array.array(ty)
    a.frombytes(s[4:4+n])
    return a, s[4+n:]


if __name__ == "__main__":
    if len(sys.argv) <= 1:
        print("Usage: %s test" % sys.argv[0])
    elif sys.argv[1] == 'test':
        def test_equal(a, b):
            if a != b:
                print("Test failed: {} != {}".format(a,b))
        def test_true(a, m):
            if not a:
                print("Test failed: {}".format(m))

        test_equal(pos2_flip((11,2)), (16,25))
        test_equal(pos2_encode((13,0)), 0)
        test_equal(pos2_encode((14,0)), 1)
        test_equal(pos2_encode((12,1)), 2)
        test_equal(pos2_encode((27,13)), HALF_ARENA_VOL - 1)
        test_equal(pos2_encode((0,14)), HALF_ARENA_VOL)
        test_equal(pos2_encode((14,27)), ARENA_VOL - 1)
        test_equal(pos2_encode(pos2_flip((12,1))), ARENA_VOL - 1 - 2)
        test_equal(pos2_decode(0), (13,0))
        test_equal(pos2_decode(HALF_ARENA_VOL), (0,14))
        test_equal(pos2_decode(ARENA_VOL - 1), pos2_flip((13,0)))

        test_equal(pos2_edge_decode(0), [0,13])
        test_equal(pos2_edge_decode(13), [13,0])
        test_equal(pos2_edge_decode(3), [3, 10])
        test_equal(pos2_edge_decode(ARENA_SIZE - 1), [27,13])

        array_1 = array.array('i', [1,2,3,4])
        test_equal(array_from_string(array_to_string(array_1))[0],
                array.array('i', [1,2,3,4]))
        array_2 = array.array('i', [5,6,7,8])
        s1 = array_to_string(array_1)
        s2 = array_to_string(array_2)
        s = s1 + s2
        ra1, rs1 = array_from_string(s)
        ra2, rs2 = array_from_string(rs1)
        test_equal(ra1, array.array('i', [1,2,3,4]))
        test_equal(ra2, array.array('i', [5,6,7,8]))

        test_equal(pos2_edge_on((0,13)), 1)
        test_equal(pos2_edge_on((0,14)), 3)
        test_equal(pos2_edge_on((15,1)), 2)
        test_equal(pos2_edge_on((27,14)), 4)
        test_equal(pos2_edge_on((12,12)), 0)
        test_true(pos2_edge_isOpposing((13,0),(27,14)), "(13,0) - (27,14)")

        test_true(pos2_inbound((27,14)), "inBound 27, 14")
        test_true(pos2_inbound((13,13)), "inBound 13, 13")
        test_true(not pos2_inbound((20,2)), "!inBound 20, 2")

        print("=== Ignore the following errors ===")
        test_equal("Test1", "Test2")
        print("=== Assertion System test complete ===")

        print("Test Complete. Exit.")
    else:
        print("Unknown argument {}".format(sys.argv[1]))
