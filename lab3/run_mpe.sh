#!/bin/bash

export LD_LIBRARY_PATH=~/opp_labs/lab3/lib/:~/mpe/sw/lib/:$LD_LIBRARY_PATH
mpecc -mpilog ./src/main.cpp ./src/MatrixMultiplier.cpp -o ./bin/lab3mpe.out

qsub ./submit_mpe.sh

echo "Job submit_mpe.sh submitted with MPE"
