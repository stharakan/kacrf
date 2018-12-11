#!/bin/bash

omp=32
spa_bws=(1.0)
app_bws_spa=(20.0 40.0)
app_bws_int=(0.25 0.5 1.0)

for spa_bw in ${spa_bws[@]}
do
	for app_bw_spa in ${app_bws_spa[@]}
	do
		for app_bw_int in ${app_bws_int[@]}
		do
			# set file name
			oname=cvcrf.s${spa_bw}.as${app_bw_spa}.ai${app_bw_int}.o
			echo ${oname}

			# call func
			export OMP_NUM_THREADS=$omp
			./../bin/cvcrf ${spa_bw} ${app_bw_spa} ${app_bw_int} > ./${oname}
		done # s loop
	done # k loop
done # m loop

