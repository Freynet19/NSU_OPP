#!/bin/bash
#PBS -l walltime=00:20:00
#PBS -l select=1:ncpus=8:mpiprocs=1
#PBS -m n

cd $PBS_O_WORKDIR
MPI_NP=$(wc -l $PBS_NODEFILE | awk '{ print $1 }')
echo "Number of MPI processes: $MPI_NP"
echo "Node list:"
cat $PBS_NODEFILE
echo "----------------------------------------------------------------"

mpirun -machinefile $PBS_NODEFILE -np $MPI_NP ./bin/lab1.out

