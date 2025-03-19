#!/bin/bash

source /opt/intel/composerxe/bin/compilervars.sh intel64

mpecc -mpilog src/main.cpp src/sleSolver.cpp src/BinIO.cpp -o ./bin/lab1mpe.out

qsub ./run_scripts/lab1_run_mpe.sh

