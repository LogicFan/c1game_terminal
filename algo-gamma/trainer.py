import multiprocessing as mp
import subprocess
import shutil
import os, sys, signal, time, glob, argparse
import math, numpy, tensorflow

sys.path.insert(0, './algo')

import model, transform

os.environ['XLA_FLAGS']='--xla_hlo_profile'

lock_print = mp.Lock()
queue_return = mp.Queue()
def _worker_print(s):
    s = "%s%s " % (s, os.linesep)
    with lock_print:
        sys.stdout.write(s)

WORKER_KEY_HISTORY = "history"   # Historical versions (folder) of trainee
WORKER_KEY_CURRENT = "current"   # Current newest version (folder) of trainee
WORKER_KEY_ROUNDS = "rounds"     # Number of rounds for this worker
WORKER_KEY_ENGINE = "engine"     # Game engine
WORKER_KEY_REPLAYPY = "replay.py"     # Game engine
WORKER_KEY_TRAINEE = "trainee"   # Trainee script
WORKER_KEY_HISTORIC = "historic"   # Historic Trainee script
WORKER_KEY_TRAINERS = "trainers" # Trainer algorithms (reference)
WORKER_KEY_DRYRUN = "dryrun"
WORKER_KEY_VERBOSE = "verbose"
WORKER_KEY_GENERATION = "gen"

ENV_KEY_TRAINEE = "DIRICHLET_TRAINEE"
ENV_KEY_TRAINER = "DIRICHLET_TRAINER"

workers_list = []

def _nWorkers_default():
    return max(1, mp.cpu_count() - 2)

def _handler_sigint(sig, frame):
    print("Received Interrupt ... Terminating")
    for p in workers_list:
        p.terminate()
        p.join()
    sys.exit(0)

def _worker(dir_worker, config):
    history = config[WORKER_KEY_HISTORY]
    current = config[WORKER_KEY_CURRENT]
    rounds = config[WORKER_KEY_ROUNDS]
    engine = config[WORKER_KEY_ENGINE]
    replay_py = config[WORKER_KEY_REPLAYPY]
    trainee = config[WORKER_KEY_TRAINEE]
    historic = config[WORKER_KEY_HISTORIC]
    trainers = config[WORKER_KEY_TRAINERS]
    dryrun = config[WORKER_KEY_DRYRUN]
    verbose = config[WORKER_KEY_VERBOSE]
    generation = config[WORKER_KEY_GENERATION]

    file_stdout = open(dir_worker + "/stdout.log", "w")
    file_stderr = open(dir_worker + "/stderr.log", "w")

    # Create environment
    env = os.environ.copy()
    env[ENV_KEY_TRAINEE] = current + '/weights'

    uid = dir_worker[-2:]
    tracker = 0
    if verbose:
        _worker_print("[Worker {}] Directory = {}".format(uid, dir_worker))
    for i in range(rounds):
        # Select from history or trainee
        #opponent_id = numpy.random.randint(0, len(history) + len(trainers))
        def select_player1():
            opponent_id = numpy.random.randint(0, len(trainers) + len(history))
            if opponent_id < len(trainers):
                opponent_scr = trainers[opponent_id]
                opponent_weights = ""
            else:
                opponent_id -= len(trainers)
                #assert opponent_id < len(history)
                opponent_scr = trainee
                opponent_weights = history[-1]
            return opponent_scr, opponent_weights
        def select_player2(p1s):
            opponent_id = numpy.random.randint(0, len(history) + len(trainers))
            if opponent_id < len(trainers):
                opponent_scr = trainers[opponent_id]
                opponent_weights = ""
                if opponent_scr == p1s:
                    opponent_scr = historic
                    opponent_weights = history[-1]
            else:
                opponent_id -= len(trainers)
                assert opponent_id < len(history)
                opponent_scr = historic
                opponent_weights = history[opponent_id]
            return opponent_scr, opponent_weights
        p1s, p1w = select_player1()
        p2s, p2w = select_player2(p1s)
        env[ENV_KEY_TRAINEE] = p1w + '/weights'
        env[ENV_KEY_TRAINER] = p2w + '/weights'
        command = engine + [p1s, p2s]

        if dryrun:
            _worker_print(dir_worker[-2:] + ": " + " ".join(command))
            with open(dir_worker + "/m.{:04d}".format(i), 'w') as f:
                f.write(" ".join(command))
        else:
            if verbose:
                _worker_print(command)
            subprocess.run(command, env=env, cwd=dir_worker,
                    stdout=file_stdout, stderr=file_stderr)

            replays = glob.glob(dir_worker + "/replays/*.replay")
            for r in replays:
                if not os.path.isfile(r):
                    _worker_print("[Worker {}] Failed to find replay: {}".format(uid, r))
                    continue
                comm_convert = ["python3", replay_py, "export",
                        r, dir_worker + "/m.{:04d}.{:04d}".format(generation, tracker)]
                if verbose:
                    _worker_print(comm_convert)
                subprocess.run(comm_convert, stdout=file_stdout,
                        stderr=file_stderr)
                os.remove(r)
                tracker += 1


    _worker_print("[Worker {}] Exit. {} samples generated".format(
        uid, tracker))


class Trainer:
    
    def __init__(self):
        self.nWorkers = 1
        self.nRounds = 4
        self.perGenerationEpochs = 4
        self.perGenerationBatch = 32
        self.learningRate = 0.001
        self.dryrun = False
        self.verbose = False

        # Directory for worker threads
        self.dir_base = os.getcwd()
        self.dir_train = self.dir_base + "/workers"
        # Directory to store historical models
        self.dir_history = self.dir_base + "/history"
        # Directory for the engine
        self.dir_kit = self.dir_base + "/kit"
        self.dir_replaypy = self.dir_base + "/trainlib/replay.py"
        self.scr_engine = ['java', '-jar',
                self.dir_kit + '/engine.jar', 'work']
        # script for the initial training algorithm (non-neural)
        self.scr_trainers = [
                self.dir_kit + "/python-algo/run.sh",
                self.dir_base + "/logicfan/python-algo/run.sh",
                self.dir_base + "/logicfan/python-algo-blackbeard/run.sh",
                self.dir_base + "/algo-trainers/random.run.sh",
            ]
        # Script for the main NN based algorithm (trainee)
        self.scr_historic =  os.getcwd() + "/algo/run-trainer.sh"
        self.scr_trainee =  os.getcwd() + "/algo/run-trainee.sh"
        self.historyEntries = []
        self.PREFIX_HISTORY = 'g'
        self.PREFIX_WORKER = 'w'
        self.nextGeneration = 0


    def printConfig(self):
        print("Worker threads: %s" % self.nWorkers)
        print("Learning Rate: {}".format(self.learningRate))
        print("dir_train: %s"      % self.dir_train)
        print("dir_hist: %s"       % self.dir_history)
        print("dir_kit: %s"        % self.dir_kit)
        print("dir_replaypy: %s"   % self.dir_replaypy)
        print("scr_engine: %s"     % self.scr_engine)
        print("scr_trainers: %s"   % self.scr_trainers)
        print("scr_historic: %s"   % self.scr_historic)
        print("scr_trainee: %s"    % self.scr_trainee)
        print("Historical Entries: {}".format(self.historyEntries))
        print("Next generation number: {}".format(self.nextGeneration))

    def loadHistory(self):
        if not os.path.isdir(self.dir_train):
            print("Creating training dir: {}".format(self.dir_train))
            os.makedirs(self.dir_train)
        for i in range(self.nWorkers):
            dir_worker = self.getWorkerDirectory(i)
            if not os.path.isdir(dir_worker):
                os.makedirs(dir_worker)
                print("Worker {} directory created: {}".format(i, dir_worker))
            else:
                print("Worker {} directory exist: {}".format(i, dir_worker))
        if not os.path.isdir(self.dir_history):
            os.makedirs(self.dir_history)

        for entry in os.scandir(self.dir_history):
            if not os.path.isdir(entry):
                ccurrentontinue
            if not entry.name.startswith(self.PREFIX_HISTORY):
                continue
            # A history exists
            self.historyEntries.append(self.dir_history + "/" + entry.name)
            entry_gen = int(entry.name[len(self.PREFIX_HISTORY):])
            if self.nextGeneration <= entry_gen:
                self.nextGeneration = entry_gen + 1
        self.historyEntries.sort()

            #print(entry.name)

    def getWorkerDirectory(self, i: int):
        return self.dir_train + "/{}{:02d}".format(self.PREFIX_WORKER, i)
    def getGenerationDirectory(self, i: int):
        return self.dir_history + "/{}{:03d}".format(self.PREFIX_HISTORY, i)

    def generateTrainingData(self):
        print("=== Generation {} ===".format(self.nextGeneration))
        if not self.historyEntries:
            print("The initial generation has to be trained first.")
            return False

        if self.dryrun:
            print("This is a dry run!")

        config = {
            WORKER_KEY_HISTORY  : self.historyEntries,
            WORKER_KEY_CURRENT  : self.historyEntries[-1],
            WORKER_KEY_ROUNDS   : self.nRounds // self.nWorkers,
            WORKER_KEY_ENGINE   : self.scr_engine,
            WORKER_KEY_REPLAYPY : self.dir_replaypy,
            WORKER_KEY_TRAINEE  : self.scr_trainee,
            WORKER_KEY_HISTORIC : self.scr_historic,
            WORKER_KEY_TRAINERS : self.scr_trainers,
            WORKER_KEY_DRYRUN   : self.dryrun,
            WORKER_KEY_VERBOSE  : self.verbose,
            WORKER_KEY_GENERATION : self.nextGeneration,
        }
        workers_list = []
        for i in range(self.nWorkers):
            dir_worker = self.getWorkerDirectory(i)
            if not os.path.isfile(dir_worker + "/game-configs.json"):
                shutil.copy2(self.dir_kit + "/game-configs.json", dir_worker)

            p = mp.Process(target=_worker, args=(dir_worker, config))
            p.start()

            workers_list.append(p)
        for p in workers_list:
            p.join()
        print("Generation {} trained for {} rounds".format(
            self.nextGeneration,
            self.nRounds))

        nFiles = 0
        for i in range(self.nWorkers):
            dir_worker = self.getWorkerDirectory(i)
            for filename in os.listdir(dir_worker):
                if not filename.endswith(".a"):
                    continue
                else:
                    nFiles += 1

        if self.dryrun:
            print("Generated {} data".format(nFiles))
        return True

    def loadSamples(self):
        """
        Read all samples from worker directories and concatenate them
        """

        pattern = self.dir_train + "/" + self.PREFIX_WORKER + "*/*.in.V.npy"
        files = glob.glob(pattern)
        if files:
            multi_inV, multi_inS, multi_outV1, multi_outV2, multi_weightV1, multi_weightV2 = \
                    zip(*[model.loadSample(f[:-9]) for f in files])
            return numpy.concatenate(multi_inV), numpy.concatenate(multi_inS), \
                   numpy.concatenate(multi_outV1), numpy.concatenate(multi_outV2),\
                   numpy.concatenate(multi_weightV1),numpy.concatenate(multi_weightV2)
        else:
            return None

    def trainGeneration(self):
        m = model.create()
        opt = tensorflow.keras.optimizers.Adam(
                learning_rate=self.learningRate)
        #opt = tensorflow.keras.optimizers.SGD(
        #        learning_rate=self.learningRate,
        #        decay=1e-6,
        #        momentum=0.9,
        #        nesterov=True)

        HP_V1_WEIGHT = 1
        HP_V2_WEIGHT = 1
        print("w_1 = {}".format(HP_V1_WEIGHT))
        print("w_2 = {}".format(HP_V2_WEIGHT))

        lossV1 = tensorflow.keras.losses.Huber(delta=0.03)
        lossV2 = tensorflow.keras.losses.Huber(delta=0.03)
        #lossV1 = "mean_squared_error"
        #lossV2 = "mean_squared_error"

        m.compile(
            optimizer=opt,
            loss={'V1': lossV1, 'V2': lossV2},
            loss_weights={'V1': HP_V1_WEIGHT, 'V2': HP_V2_WEIGHT},
            #loss="logcosh",
            metrics=['accuracy']
        )
        path_weight = self.historyEntries[-1] + "/weights"
        try:
            m.load_weights(path_weight)
        except:
            print("Failed to load weights from file: " + path_weight)

        print("Loading Samples ...")
        samples = self.loadSamples()

        if samples:
            trainIV,trainIS,trainOV1,trainOV2,weightsV1,weightsV2 = samples

            print("trainIV: {}".format(trainIV.shape))
            print("trainIS: {}".format(trainIS.shape))
            print("trainOV1: {}".format(trainOV1.shape))
            print("trainOV2: {}".format(trainOV2.shape))
            print("weightsV1: {}".format(weightsV1.shape))
            print("weightsV2: {}".format(weightsV2.shape))

            print("Weights V1 (head): {}".format(weightsV1[:10]))
            print("Weights V2 (head): {}".format(weightsV2[:10]))

            # Main training function
            m.fit({'iV': trainIV, 'iS': trainIS},
                  {'V1': trainOV1, 'V2': trainOV2},
                  epochs=self.perGenerationEpochs,
                  batch_size=self.perGenerationBatch,
                  sample_weight={'V1': weightsV1, 'V2': weightsV2}
            )

            nextGenDir = self.getGenerationDirectory(self.nextGeneration)
            if not os.path.isdir(nextGenDir):
                os.makedirs(nextGenDir)
            self.historyEntries.append(nextGenDir)
            path_weight_next = nextGenDir + "/weights"
            m.save_weights(path_weight_next)
            print("Trained weights (gen={}): {} -> {}".format(
                self.nextGeneration - 1, path_weight, path_weight_next))
            return True
        else:
            print("Unable to load samples!")
            return False


    def train(self, generations: int):
        for i in range(generations):
            self.generateTrainingData()
            flag = self.trainGeneration()
            if flag:
                self.nextGeneration += 1
            else:
                print("Failed to train: {}".format(self.nextGeneration))
        print("Training Complete. Generations={}".format(generations))




if __name__ == "__main__":
    signal.signal(signal.SIGINT, _handler_sigint)

    parser = argparse.ArgumentParser(description="Trainer")
    parser.add_argument('-D', dest='dryrun', action='store_true',
                        default=False, help='Dryrun')
    parser.add_argument('-V', dest='verbose', action='store_true',
                        default=False, help='Verbose')
    parser.add_argument('-G', dest='dataonly', action='store_true',
                        default=False, help='Generate data only')
    parser.add_argument('-o', dest='trainonly', action='store_true',
                        default=False, help='Train Only')
    parser.add_argument('-w', dest='nWorkers', action='store',
                        type=int, default=2, help='Set to 0 for automatic')
    parser.add_argument('-r', dest='nRounds', action='store',
                        type=int, default=4)
    parser.add_argument('-e', dest='nEpochs', action='store',
                        type=int, default=4)
    parser.add_argument('-b', dest='batchSize', action='store',
                        type=int, default=32)
    parser.add_argument('-g', dest='generations', action='store',
                        type=int, default=1)
    parser.add_argument('-l', dest='learningRate', action='store',
                        type=float, default=0.001)

    args = parser.parse_args()

    trainer = Trainer()

    trainer.dryrun = args.dryrun
    trainer.verbose = args.verbose
    if args.nWorkers == 0:
        trainer.nWorkers = nWorkers_default()
    else:
        trainer.nWorkers = args.nWorkers
    trainer.nRounds = args.nRounds
    trainer.perGenerationEpochs = args.nEpochs
    trainer.perGenerationBatch = args.batchSize
    trainer.learningRate = args.learningRate

    trainer.loadHistory()
    trainer.printConfig()

    # Train for g generations
    if args.generations <= 0:
        print("Number of generations must be positive!")
    elif args.dataonly:
        trainer.generateTrainingData()
    elif args.generations == 1:
        if not args.trainonly:
            trainer.generateTrainingData()
        trainer.trainGeneration()
    else:
        trainer.train(args.generations)
