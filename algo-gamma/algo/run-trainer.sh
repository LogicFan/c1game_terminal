#!/bin/bash

DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
export CUDA_VISIBLE_DEVICES='-1'
${PYTHON_CMD:-python3} -u "$DIR/algo_strategy.py" $DIRICHLET_TRAINER
