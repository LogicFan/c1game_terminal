import os, sys
import argparse
import subprocess

if not os.path.isdir("kit"):
    print("source path does not exist!")
    print("Please create a symlink using the following command")
    print("")
    print("\t $ ln -s <C1GamesStarterKit-master-path> kit")
    sys.exit()

PATH_KIT="kit"

parser = argparse.ArgumentParser(description='Dirichlet Algorithm Trainer')

subparsers =  parser.add_subparsers(dest='MODE')
parser_test = subparsers.add_parser('test')
parser_train = subparsers.add_parser('train')
parser_deploy  = subparsers.add_parser('deploy')
#parser.add_argument('--sum', dest='accumulate', action='store_const',
#                    const=sum, default=max,
#                    help='sum the integers (default: find the max)')

args = parser.parse_args()

print("Mode = {}".format(args.MODE))

print(str(args))
if args.MODE == 'test':
    print("Testing...")
    subprocess.call(["python3", "algo/algo_strategy.py", "test"])
elif args.MODE == 'deploy':
    if not os.path.isdir("build"):
        os.makedirs("build")
    subprocess.call(["zip", "build/Dirichlet.zip", "-r", "algo"])
else:
    print("Unknown Mode: {}".format(args.MODE))
