#!/bin/bash
rm out/*
rm err/*
for i in 0 1 2 3;
do
for k in 0 1 3;
do
	sbatch -p opteron main.mpi $i $k
	sbatch -p opteron main.mpi $i $k
done;
done;
