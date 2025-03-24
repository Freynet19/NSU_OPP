#!/bin/bash

if [ $# -ne 1 ] || [[ ! $1 =~ ^(1|2|4|8|16)$ ]]; then
    echo "Argument must be one of these: 1, 2, 4, 8, 16" >&2
    exit 1
fi

source /opt/intel/composerxe/bin/compilervars.sh intel64
mpiicpc -O3 ./src/main.cpp ./src/sleSolver.cpp ./src/BinIO.cpp -o ./bin/lab1.out

qsub ./run_scripts/lab1_run_$1p.sh

