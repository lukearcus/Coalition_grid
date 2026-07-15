#!/bin/bash
#PBS -l select=1:ncpu=2:mem=4gb
#PBS -l walltime=00:00:01
#PBS -N coal_grid

module load Julia

cd $PBS_O_WORKDIR

julia test.jl
