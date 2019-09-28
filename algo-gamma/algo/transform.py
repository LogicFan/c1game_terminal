import math, sys

ARENA_SIZE = 28
HALF_ARENA = int(ARENA_SIZE / 2)
HALF_ARENA_VOL = 210
ARENA_VOL = HALF_ARENA_VOL * 2

# Operations that transform the game state into a input vector.

def pos2_flip(pos):
    x, y = pos
    assert 0 <= x and x <= 27
    assert 0 <= y and y <= 27
    return (27 - x, 27 - y)
def pos2_encode(pos):
    """
    Converts a arena position (x,y) into a serialised position n
    """
    assert 0 <= pos[1] and pos[1] <= 27
    def pos2_encode_lower(pos):
        x, y = pos
        assert 0 <= x and x <= 27
        assert 0 <= y and y <= 13
        left_x = 13 - y
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
        left_x = 13 - y
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
    if n >= 14:
        return [n, (n - 14)]
    else:
        return [n, (13 - n)]

if __name__ == "__main__":
    if len(sys.argv) <= 1:
        print("Usage: %s test" % sys.argv[0])
    elif sys.argv[1] == 'test':
        def test_equal(a, b):
            if a != b:
                print("Test failed: {} != {}".format(a,b))
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

        print("=== Ignore the following errors ===")
        test_equal("Test1", "Test2")
        print("=== Assertion System test complete ===")

        print("Test Complete. Exit.")
    else:
        print("Unknown argument {}".format(sys.argv[1]))
