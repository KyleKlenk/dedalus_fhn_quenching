#!/bin/bash
index=10
N=4096
L=1500
soldir=./waves/index_$((index))/
mkdir -p $soldir
echo "N=$N and L=$L, base directory: $soldir"
python3 waves.py --N=$N --L=$L --index=$index --soldir=$soldir --autodir=./auto/rg &

index=11
N=4096
L=2700
soldir=./waves/index_$((index))/
mkdir -p $soldir
echo "N=$N and L=$L, base directory: $soldir"
python3 waves.py --N=$N --L=$L --index=$index --soldir=$soldir --autodir=./auto/rg &
