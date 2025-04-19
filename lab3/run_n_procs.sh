#!/bin/bash

valid_procs=(1 2 3 4 5 6 7 8 10 12 14 16)
if [ $# -ne 1 ] || [[ ! " ${valid_procs[@]} " =~ " $1 " ]]; then
    echo "Usage: $0 <num_procs>"
    echo "Valid values: ${valid_procs[*]}" >&2
    exit 1
fi

export LD_LIBRARY_PATH=~/opp_labs/lab3/lib/:$LD_LIBRARY_PATH
mpiicpc -O3 ./src/main.cpp ./src/MatrixMultiplier.cpp -o ./bin/lab3.out

nodes=1
procs_per_node=$1

if [[ $1 -gt 8 ]]; then
    nodes=2
    procs_per_node=$(( $1 / 2 ))
fi

TMP_SCRIPT=$(mktemp)
cat > $TMP_SCRIPT <<EOF
#!/bin/bash
#PBS -l walltime=00:30:00
#PBS -l select=${nodes}:ncpus=8:mpiprocs=${procs_per_node}
#PBS -m n

cd \$PBS_O_WORKDIR
MPI_NP=\$(wc -l \$PBS_NODEFILE | awk '{ print \$1 }')
echo "Total MPI processes: \$MPI_NP"
echo "Nodes: ${nodes}, Procs per node: ${procs_per_node}"
echo "Node list:"
cat \$PBS_NODEFILE
echo "----------------------------------------------"

matrix_sizes=(1000 1500 2000 2500)
for size in "\${matrix_sizes[@]}"; do
    echo "Running for matrix sizes: \${size}x\${size}, \${size}x\${size}"
    mpirun -machinefile \$PBS_NODEFILE -np \$MPI_NP ./bin/lab3.out \${size} \${size} \${size}
    echo "----------------------------------------------"
    sleep 1
done
EOF

qsub $TMP_SCRIPT
rm -f $TMP_SCRIPT

echo "Job $TMP_SCRIPT submitted for $1 processes"
