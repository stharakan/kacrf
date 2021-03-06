#!/bin/bash

omp=10
N=40000
Ms=(4096)
Ks=(128)
Ss=(4096)
stol=(0.000001)
Bs=(0.9)


for m in ${Ms[@]}
do
	for k in ${Ks[@]}
	do
		for s in ${Ss[@]}
		do
			for budget in ${Bs[@]}
			do
				# set file name
				oname=kspa.td${omp}.n${N}.m${m}.k${k}.s${s}.t${stol}.b${budget}.o
				oname2=kspa.td${omp}.n${N}.m${m}.k${k}.s${s}.t${stol}.b${budget}.m
				echo ${oname}

				# call func
				export OMP_NUM_THREADS=$omp
				./../bin/kspa-test ${N} ${m} ${k} ${s} ${stol} ${budget} > ./${oname}

				# move interaction mat
				cp interaction.m ${oname2}
			done # budget loop
		done # s loop
	done # k loop
done # m loop

