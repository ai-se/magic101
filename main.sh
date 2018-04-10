#!/bin/bash
rm out/*
rm err/*
for i in 0 1 2 3 4 5 6 7 8;
do
for k in 5;
do
	sbatch -p opteron main.mpi $i $k
	sbatch -p opteron main.mpi $i $k
done;
done;
