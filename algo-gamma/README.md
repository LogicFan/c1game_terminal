# Dirichlet, A Terminal AI

## Build and Training

A symbolic link must be created to `C1GamesStarterKit-master`, via the
command
```
$ ln -s <C1GamesStarterKit-master-path> kit
```

To deploy the algorithm for production:
```
python3 main.py deploy
```

### Files of interest

```
Dirichlet
 |
 |- main.py # Deployment and testing
 |- trainer.py # Training
 |- algo # Main algorithms folder
 |  |
 |  |- model.py     # Contains the keras model
 |  |- transform.py # Position encoding
 |  |- visual.py    # State Visualisation
 |  |- run.sh       # Production run script
 |  |- run-trainee.sh # Training run script (p1)
 |  |- run-trainer.sh # Trainer run script (p1)
 |
 |- algo-trainers # Training algorithms
    |
    |  # A randomised algorithm
    |- random.agent.py
    |- random.run.{ps1,sh}
```


## AI Architecture

### Preprocessing

Since the game board is a rhombus, it has to be rotated into a square before we
can feed the game state to the neural network. Let N be the edge length of the
rhombus. 

### Neural Network Specification

Since the game is close to full information, we do not need to remember
the game state from previous states (as opposed to games like Dota 2).

The inputs are

| Shape   | Range   | Source                    |
| ------- | ------- | ------------------------- |
| Board   | [0,1]   | Filter Health (S&E)       |
| Board   | [0,1]   | Destructor Heblth (S&E)   |
| Board   | [0,1]   | Encryptor Health (S&E)    |
| Board   | {0,1}   | Filter Unit (S&E)         |
| Board   | {0,1}   | Destructor Unit (S&E)     |
| Board   | {0,1}   | Encryptor Unit (S&E)      |
| 1       | [0,1]   | Self Health               |
| 1       | [0,->]  | Self Bits/5               |
| 1       | [0,->]  | Self Cores/5              |
| 1       | [0,1]   | Enemy Health              |
| 1       | [0,->]  | Enemy Bits/5              |
| 1       | [0,->]  | Enemy Cores/5             |
| 1       | [0,->]  | Turn number/10            |

The outputs are

| Shape   | Range  | Description                    |
| ------- | ------ | ------------------------------ |
| Tri     | [0,1]  | Deletion Policy                |
| Tri     | [0,1]  | Placement of Filter            |
| Tri     | [0,1]  | Placement of Destructor        |
| Tri     | [0,1]  | Placement of Encryptor         |
| Edge    | [0,->]  | Placement of Ping              |
| Edge    | [0,->]  | Placement of EMP               |
| Edge    | [0,->]  | Placement of Scrambler         |

Note: The Action Probability Vector is a distribution from which we sample the
next move. If a action is not feasible, it is not sampled, with the exception
of deleting units.
* End Turn
* Delete Unit
* Place Filter
* Place Destructor
* Place Encryptor
* Place Ping
* Place EMP
* Place Scrambler


