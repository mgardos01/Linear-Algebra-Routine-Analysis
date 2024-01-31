#!/bin/bash
git pull && make clean && make test && sbatch ./slurm.sh
while true
do 
    squeue | grep 'mgardos'
    sleep 5
done
