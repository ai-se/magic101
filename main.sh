#!/bin/bash
rm out/*
rm err/*
for i in 7;
do
for k in 5;
do
	sbatch -p opteron main.mpi $i $k
	sbatch -p opteron main.mpi $i $k
done;
done;
