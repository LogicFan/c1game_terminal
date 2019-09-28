import sys, os
import json
import math
import copy
import numpy

import model, transform
import unit
from unit import GameUnit 
import game_map
from game_map import GameMap 
from config import CONFIG




class Replay:
    def __init__(self,replay_path, config=CONFIG):
        global FILTER, ENCRYPTOR, DESTRUCTOR, PING, EMP, SCRAMBLER, REMOVE, FIREWALL_TYPES, ALL_UNITS, UNIT_TYPE_TO_INDEX, UNIT_MAPPING
        UNIT_TYPE_TO_INDEX = {}
        FILTER = config["unitInformation"][0]["shorthand"]
        UNIT_TYPE_TO_INDEX[FILTER] = 0
        ENCRYPTOR = config["unitInformation"][1]["shorthand"]
        UNIT_TYPE_TO_INDEX[ENCRYPTOR] = 1
        DESTRUCTOR = config["unitInformation"][2]["shorthand"]
        UNIT_TYPE_TO_INDEX[DESTRUCTOR] = 2
        PING = config["unitInformation"][3]["shorthand"]
        UNIT_TYPE_TO_INDEX[PING] = 3
        EMP = config["unitInformation"][4]["shorthand"]
        UNIT_TYPE_TO_INDEX[EMP] = 4
        SCRAMBLER = config["unitInformation"][5]["shorthand"]
        UNIT_TYPE_TO_INDEX[SCRAMBLER] = 5
        REMOVE = config["unitInformation"][6]["shorthand"]
        UNIT_TYPE_TO_INDEX[REMOVE] = 6
        UNIT_MAPPING=[FILTER,ENCRYPTOR,DESTRUCTOR,PING,EMP,SCRAMBLER,REMOVE]
        ALL_UNITS = [PING, EMP, SCRAMBLER, FILTER, ENCRYPTOR, DESTRUCTOR]
        FIREWALL_TYPES = [FILTER, ENCRYPTOR, DESTRUCTOR]

        self.ARENA_SIZE = 28
        self.HALF_ARENA = int(self.ARENA_SIZE / 2)
        self.BITS = 0
        self.CORES = 1
        self.config = config
        self.replay_path=replay_path
        def replay_state(replay_path):
            with open(replay_path) as f:
                data = f.readlines()
            index=0
            initialstate=[]
            accu=[]
            start=True
            while(index<len(data)):
                def repl(x):
                    return x.replace('false', '"False"')
                #if data[index].replace('false','"False"').find("-1")!=-1:
                if repl(data[index]).find("-1")!=-1:
                    if start:
                        initialstate=repl(data[index+1])
                        start=False
                    else:
                        endstate=repl(data[index-1])
                        accu.append([eval(initialstate)]+[eval(endstate)])
                        initialstate=repl(data[index+1])
                index=index+1
            return accu
        self.replay=replay_state(self.replay_path)
        self.replay_result=self.parse_replay()

    def parse_replay_dict(self,index,turn_number):
        result={"p1map":[],"p2map":[],"p1Stats":[],"p2Stats":[]}
        default1=GameMap(self.config)
        default2=GameMap(self.config)
        for i in range(len(self.replay[turn_number][index]['p1Units'])):
            for j in self.replay[turn_number][index]['p1Units'][i]:
                default1[j[0],j[1]].append(GameUnit(UNIT_MAPPING[i],self.config,0,j[2],j[0],j[1]))
        for i in range(len(self.replay[turn_number][index]['p2Units'])):
            for j in self.replay[turn_number][index]['p2Units'][i]:
                default2[j[0],j[1]].append(GameUnit(UNIT_MAPPING[i],self.config,1,j[2],j[0],j[1]))
        result["p1map"]=copy.deepcopy(default1)
        result["p2map"]=copy.deepcopy(default2)
        result["p1Stats"]=self.replay[turn_number][index]["p1Stats"]
        result["p2Stats"]=self.replay[turn_number][index]["p2Stats"]
        result["turnInfo"]=self.replay[turn_number][index]["turnInfo"][0]
        return result
    
    def parse_replay(self):
        accu=[]
        for i in range(0,len(self.replay)):
            accu.append([self.parse_replay_dict(0,i),self.parse_replay_dict(1,i)])
        return accu
    
    def get_unit_list(self,turn_number,x,y,state=1,player=0):
        if player==0:
            map_state='p1map'
        else:
            map_state='p2map'
        return self.replay_result[turn_number-1][state][map_state][x,y]
    def get_player_health(self,turn_number,state=1,player=0):
        if player==0:
            map_state='p1Stats'
        else:
            map_state='p2Stats'
        return self.replay_result[turn_number-1][state][map_state][0]
    def get_player_bit(self,turn_number,state=1,player=0):
        if player==0:
            map_state='p1Stats'
        else:
            map_state='p2Stats'
        return self.replay_result[turn_number-1][state][map_state][1]
    def get_player_core(self,turn_number,state=1,player=0):
        if player==0:
            map_state='p1Stats'
        else:
            map_state='p2Stats'
        return self.replay_result[turn_number-1][state][map_state][2]
    def get_total_turn_number(self):
        return len(self.replay_result)
    def get_new_units(self,turn_number,x,y,player=0):
        if turn_number==1:
            return self.get_unit_list(1,x,y,0,player)
        else:
            return list(set(self.get_unit_list(turn_number,x,y,0,player))
                        -set(self.get_unit_list(turn_number-1,x,y,1,player)))
    def get_removed_units(self,turn_number,x,y,player=0):
        if turn_number==1:
            return []
        else:
            return list(set(self.get_unit_list(turn_number-1,x,y,1,player))
                        -set(self.get_unit_list(turn_number,x,y,0,player)))
    def get_winner(self):
        p1hp=self.get_player_health(self.get_total_turn_number(),0)
        p2hp=self.get_player_health(self.get_total_turn_number(),1)
        if p1hp>p2hp:
            return 0
        else:
            return 1

    def get_bit_biased(self, turns,player=0):
        base = self.get_player_bit(turns,state=1,player=0)
        if turns == 1:
            return base
        return base + (turns // 10) + 5
    def get_core_biased(self, turns, player=0):
        base = self.get_player_core(turns,state=1,player=0)
        if turns == 1:
            return base
        other = 1 - player
        diffhealth = self.get_player_health(turns-1,state=0,player=other) \
                   - self.get_player_health(turns-1,state=1,player=other)
        return (base + (turns // 10) + 5 + diffhealth) * 0.75

    def export(self, path, dryrun=False, dryrun_turn=0, dryrun_flip=False):
        """
        Creates the following files:
        $path.{A,B}.in.V
        $path.{A,B}.in.S
        $path.{A,B}.out.V1
        $path.{A,B}.out.V2
        """
        path_in_V   = path + ".in.V"
        path_in_S   = path + ".in.S"
        path_A_out_V1 = path + ".A.out.V1"
        path_A_out_V2 = path + ".A.out.V2"
        path_B_out_V1 = path + ".B.out.V1"
        path_B_out_V2 = path + ".B.out.V2"

        turns = self.get_total_turn_number()

        # Last turn data to determine the winner
        player0_fhealth = self.get_player_health(turns, state=1,player=0)
        player1_fhealth = self.get_player_health(turns, state=1,player=1)

        if dryrun:
            print("Player 0 health: {}".format(player0_fhealth))
            print("Player 1 health: {}".format(player1_fhealth))

        a_inV = numpy.zeros((turns, model.IN_V_SIZE, transform.ARENA_VOL))

        # turns+1 includes last round state
        a_inS = numpy.zeros((turns+1, model.IN_S_SIZE))
        a_outV1 = numpy.zeros((turns, model.OUT_V1_SIZE, transform.HALF_ARENA_VOL))
        a_outV2 = numpy.zeros((turns, model.OUT_V2_SIZE, transform.ARENA_SIZE))
        b_outV1 = numpy.zeros((turns, model.OUT_V1_SIZE, transform.HALF_ARENA_VOL))
        b_outV2 = numpy.zeros((turns, model.OUT_V2_SIZE, transform.ARENA_SIZE))

        # Iterate through each round
        for i in range(turns):
            # inV
            for j in range(transform.ARENA_VOL):
                x,y = transform.pos2_decode(j)
                if i == 0:
                    unitlist = []
                else:
                    unitlist \
                        = self.get_unit_list(i, x, y, state=1, player=0) \
                        + self.get_unit_list(i, x, y, state=1, player=1)
                for unit in unitlist:
                    if unit.stationary:
                        health = unit.stability / unit.max_stability
                        if unit.unit_type == FILTER:
                            a_inV[i][model.IN_V_HEALTH_FILTER][j] = health
                            a_inV[i][model.IN_V_EXIST_FILTER][j] = 1
                        elif unit.unit_type == ENCRYPTOR:
                            a_inV[i][model.IN_V_HEALTH_ENCRYPTOR][j] = health
                            a_inV[i][model.IN_V_EXIST_ENCRYPTOR][j] = 1
                        elif unit.unit_type == DESTRUCTOR:
                            a_inV[i][model.IN_V_HEALTH_DESTRUCTOR][j] = health
                            a_inV[i][model.IN_V_EXIST_DESTRUCTOR][j] = 1
                        break
            # inS
            assert model.NORMAL_S_HEALTH == 30
            a_inS[i][model.IN_S_TURN] = (i+1) / model.NORMAL_S_TURN
            a_inS[i][model.IN_S_SELF_HEALTH] = \
                    self.get_player_health(i+1,state=1,player=0) / model.NORMAL_S_HEALTH
            a_inS[i][model.IN_S_SELF_BITS] = \
                    self.get_bit_biased(i+1,player=0) / model.NORMAL_S_BITS
            a_inS[i][model.IN_S_SELF_CORES] = \
                    self.get_core_biased(i+1,player=0) / model.NORMAL_S_CORES
            a_inS[i][model.IN_S_ENEMY_HEALTH] = \
                    self.get_player_health(i+1,state=1,player=1) / model.NORMAL_S_HEALTH
            a_inS[i][model.IN_S_ENEMY_BITS] = \
                    self.get_bit_biased(i+1,player=1) / model.NORMAL_S_BITS
            a_inS[i][model.IN_S_ENEMY_CORES] = \
                    self.get_core_biased(i+1,player=1) / model.NORMAL_S_CORES

            assert a_inS[i][model.IN_S_SELF_HEALTH] <= 1
            assert a_inS[i][model.IN_S_ENEMY_HEALTH] <= 1

            # outV1
            for j in range(transform.HALF_ARENA_VOL):
                # P0
                x,y = transform.pos2_decode(j)
                li = self.get_new_units(i+1, x, y, player=0)
                if a_inV[i][model.IN_V_EXIST_FILTER][j] == 1 \
                        or a_inV[i][model.IN_V_EXIST_ENCRYPTOR][j] == 1 \
                        or a_inV[i][model.IN_V_EXIST_DESTRUCTOR][j] == 1:
                    li = []
                for unit in li:
                    if unit.unit_type == FILTER:
                        a_outV1[i][model.OUT_V1_PLACE_FILTER][j] = 1
                    elif unit.unit_type == ENCRYPTOR:
                        a_outV1[i][model.OUT_V1_PLACE_ENCRYPTOR][j] = 1
                    elif unit.unit_type == DESTRUCTOR:
                        a_outV1[i][model.OUT_V1_PLACE_DESTRUCTOR][j] = 1
                if self.get_removed_units(i+1, x, y, player=0):
                    a_outV1[i][model.OUT_V1_DELETE][j] = 0 #1 SHORTCIRCUIT
                # P1
                x,y = transform.pos2_flip((x,y))
                li = self.get_new_units(i+1, x, y, player=1)
                j2 = transform.ARENA_VOL - 1 - j
                if a_inV[i][model.IN_V_EXIST_FILTER][j2] == 1 \
                        or a_inV[i][model.IN_V_EXIST_ENCRYPTOR][j2] == 1 \
                        or a_inV[i][model.IN_V_EXIST_DESTRUCTOR][j2] == 1:
                    li = []
                for unit in li:
                    if unit.unit_type == FILTER:
                        b_outV1[i][model.OUT_V1_PLACE_FILTER][j] = 1
                    elif unit.unit_type == ENCRYPTOR:
                        b_outV1[i][model.OUT_V1_PLACE_ENCRYPTOR][j] = 1
                    elif unit.unit_type == DESTRUCTOR:
                        b_outV1[i][model.OUT_V1_PLACE_DESTRUCTOR][j] = 1
                if self.get_removed_units(i+1, x, y, player=1):
                    b_outV1[i][model.OUT_V1_DELETE][j] = 0 #1 SHORTCIRCUIT
            # outV2
            for j in range(transform.ARENA_SIZE):
                x,y = transform.pos2_edge_decode(j)
                for unit in self.get_new_units(i+1, x, y, player=0):
                    if unit.unit_type == PING:
                        a_outV2[i][model.OUT_V2_PLACE_PING][j] += 1
                    elif unit.unit_type == EMP:
                        a_outV2[i][model.OUT_V2_PLACE_EMP][j] += 1
                    elif unit.unit_type == SCRAMBLER:
                        a_outV2[i][model.OUT_V2_PLACE_SCRAMBLER][j] += 1
                x,y = transform.pos2_flip((x,y))
                for unit in self.get_new_units(i+1, x, y, player=1):
                    if unit.unit_type == PING:
                        b_outV2[i][model.OUT_V2_PLACE_PING][j] += 1
                    elif unit.unit_type == EMP:
                        b_outV2[i][model.OUT_V2_PLACE_EMP][j] += 1
                    elif unit.unit_type == SCRAMBLER:
                        b_outV2[i][model.OUT_V2_PLACE_SCRAMBLER][j] += 1
        
        a_inS[turns][model.IN_S_TURN] = (turns+1) / model.NORMAL_S_TURN
        a_inS[turns][model.IN_S_SELF_HEALTH] = \
              self.get_player_health(turns,state=1,player=0) \
              / model.NORMAL_S_HEALTH
        a_inS[turns][model.IN_S_SELF_BITS] = a_inS[turns-1][model.IN_S_SELF_BITS]
        a_inS[turns][model.IN_S_SELF_CORES] = a_inS[turns-1][model.IN_S_SELF_CORES]
        a_inS[turns][model.IN_S_ENEMY_HEALTH] = \
              self.get_player_health(turns,state=1,player=1) \
              / model.NORMAL_S_HEALTH
        a_inS[turns][model.IN_S_ENEMY_BITS] = a_inS[turns-1][model.IN_S_ENEMY_BITS] 
        a_inS[turns][model.IN_S_ENEMY_CORES] = a_inS[turns-1][model.IN_S_ENEMY_CORES] 

        if dryrun:
            print("Total turns: {}".format(turns))
            print("in V: {}".format(a_inV.shape))
            print("in S: {}".format(a_inS.shape))
            print("out A V1: {}".format(a_outV1.shape))
            print("out A V2: {}".format(a_outV2.shape))
            print("out B V1: {}".format(b_outV1.shape))
            print("out B V2: {}".format(b_outV2.shape))

            T_TURN = dryrun_turn

            if dryrun_flip:
                a_inV = numpy.flip(a_inV, -1)
                a_inS = numpy.apply_along_axis(model.reverse_S, -1, a_inS)

            elems = [[' ' for x in range(transform.ARENA_SIZE)] for y in
                    range(transform.ARENA_SIZE)]
            for p in range(transform.ARENA_VOL):
                x,y = transform.pos2_decode(p)
                if a_inV[T_TURN][model.IN_V_EXIST_FILTER][p] > 0:
                    elems[y][x] = "F"
                if a_inV[T_TURN][model.IN_V_EXIST_ENCRYPTOR][p] > 0:
                    elems[y][x] = "E"
                if a_inV[T_TURN][model.IN_V_EXIST_DESTRUCTOR][p] > 0:
                    elems[y][x] = "D"


            drops = [[' ' for x in range(transform.ARENA_SIZE)] for y in
                    range(transform.ARENA_SIZE)]
            for p in range(transform.HALF_ARENA_VOL):
                x,y = transform.pos2_decode(p)
                if dryrun_flip:
                    x,y = transform.pos2_flip((x,y))
                if a_outV1[T_TURN][model.OUT_V1_PLACE_FILTER][p] > 0:
                    drops[y][x] = 'F'
                if a_outV1[T_TURN][model.OUT_V1_PLACE_ENCRYPTOR][p] > 0:
                    drops[y][x] = 'E'
                if a_outV1[T_TURN][model.OUT_V1_PLACE_DESTRUCTOR][p] > 0:
                    drops[y][x] = 'D'
                if a_outV1[T_TURN][model.OUT_V1_DELETE][p] > 0:
                    drops[y][x] = 'x'
                x,y = transform.pos2_flip((x,y))
                if b_outV1[T_TURN][model.OUT_V1_PLACE_FILTER][p] > 0:
                    drops[y][x] = 'F'
                if b_outV1[T_TURN][model.OUT_V1_PLACE_ENCRYPTOR][p] > 0:
                    drops[y][x] = 'E'
                if b_outV1[T_TURN][model.OUT_V1_PLACE_DESTRUCTOR][p] > 0:
                    drops[y][x] = 'D'
                if b_outV1[T_TURN][model.OUT_V1_DELETE][p] > 0:
                    drops[y][x] = 'x'
            for p in range(transform.ARENA_SIZE):
                x,y = transform.pos2_edge_decode(p)
                if dryrun_flip:
                    x,y = transform.pos2_flip((x,y))
                if a_outV2[T_TURN][model.OUT_V2_PLACE_PING][p] > 0:
                    drops[y][x] = 'p'
                if a_outV2[T_TURN][model.OUT_V2_PLACE_EMP][p] > 0:
                    drops[y][x] = 'e'
                if a_outV2[T_TURN][model.OUT_V2_PLACE_SCRAMBLER][p] > 0:
                    drops[y][x] = 's'
                x,y = transform.pos2_flip((x,y))
                if b_outV2[T_TURN][model.OUT_V2_PLACE_PING][p] > 0:
                    drops[y][x] = 'p'
                if b_outV2[T_TURN][model.OUT_V2_PLACE_EMP][p] > 0:
                    drops[y][x] = 'e'
                if b_outV2[T_TURN][model.OUT_V2_PLACE_SCRAMBLER][p] > 0:
                    drops[y][x] = 's'
            grid = ["[" + " ".join(e) + "] [" + " ".join(d) + "]"
                    for (e,d) in zip(elems, drops)]
            grid = "\n".join(reversed(grid))
            print("   STAGE" + " " * (transform.ARENA_SIZE*2) + "DROPS")
            print(grid)
            denormal = numpy.array([
                model.NORMAL_S_TURN,
                model.NORMAL_S_HEALTH,
                model.NORMAL_S_BITS,
                model.NORMAL_S_CORES,
                model.NORMAL_S_HEALTH,
                model.NORMAL_S_BITS,
                model.NORMAL_S_CORES,
                ])
            print("S Before: {}".format(a_inS[T_TURN] * denormal))
            print("S After: {}".format(a_inS[T_TURN+1] * denormal))

        if dryrun:
            print("numpy.save('%s', a_inV)" % path_in_V)
            print("numpy.save('%s', a_inS)" % path_in_S)
            print("numpy.save('%s', a_out_V1)" % path_A_out_V1)
            print("numpy.save('%s', a_out_V2)" % path_A_out_V2)
            print("numpy.save('%s', b_out_V1)" % path_B_out_V1)
            print("numpy.save('%s', b_out_V2)" % path_B_out_V2)
        else:
            numpy.save(path_in_V,   a_inV)
            numpy.save(path_in_S,   a_inS)
            numpy.save(path_A_out_V1, a_outV1)
            numpy.save(path_A_out_V2, a_outV2)
            numpy.save(path_B_out_V1, b_outV1)
            numpy.save(path_B_out_V2, b_outV2)

        return True




if __name__ == "__main__":
    if len(sys.argv) <= 2:
        print("Usage: %s test FILE" % sys.argv[0])
    elif sys.argv[1] == 'test':
        r = Replay(sys.argv[2])
        print("Replay object len: {}".format(len(r.replay_result)))
        print("Total turns: {}".format(r.get_total_turn_number()))
        print("Get winner: {}".format(r.get_winner()))
    elif sys.argv[1] == 'export':
        r = Replay(sys.argv[2])
        r.export(sys.argv[3])
    elif sys.argv[1] == 'dryrun':
        r = Replay(sys.argv[2])
        dryrun_turn = int(sys.argv[3])
        flip = (dryrun_turn < 0)
        r.export(sys.argv[2], dryrun=True,
                dryrun_turn=abs(dryrun_turn),
                dryrun_flip=flip)

