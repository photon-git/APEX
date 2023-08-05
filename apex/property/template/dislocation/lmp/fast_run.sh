#!/bin/bash
source activate 
a=4.03962495743116
bvector=$(echo "$a*0.7071067812" | bc -l)
repx=100
repy=140
repz=4
px=$(echo "$a*$repx*1.732050808/2-$a*0.2886751346" | bc -l)
py=$(echo "$a*$repy*0.6123724357-$a*0.2041241452" | bc -l)
pz=$(echo "$a*$repz*0.3535533905932737622004-$a*0.17677669529663688" | bc -l)
echo $px
echo $py
echo $pz
conda activate py10
lmp_mpi -var px ${px} -var py ${py} -in dislocation.in -l stroh.log
