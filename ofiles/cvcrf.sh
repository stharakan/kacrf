#!/bin/bash

# set up brain array 

declare -A high_slice

high_slice[Brats17_TCIA_620_1]=78
high_slice[Brats17_TCIA_621_1]=53
high_slice[Brats17_TCIA_623_1]=66
high_slice[Brats17_TCIA_624_1]=68
high_slice[Brats17_TCIA_625_1]=60

declare -A low_slice

low_slice[Brats17_TCIA_620_1]=69
low_slice[Brats17_TCIA_621_1]=45
low_slice[Brats17_TCIA_623_1]=49
low_slice[Brats17_TCIA_624_1]=55
low_slice[Brats17_TCIA_625_1]=50

brain=$1

hi=${high_slice[$brain]}
lo=${low_slice[$brain]}

echo $brain
echo $hi
echo $lo


omp=32
spa_bws=(1.0)
app_bws_spa=(20.0)
app_bws_spa=(5.0)
app_bws_int=(0.125 0.25 0.5 1.0)


for spa_bw in ${spa_bws[@]}
do
	for app_bw_spa in ${app_bws_spa[@]}
	do
		for app_bw_int in ${app_bws_int[@]}
		do
			# set file name
			oname_hi=${brain}.hi${hi}.${spa_bw}.as${app_bw_spa}.ai${app_bw_int}.o
			oname_lo=${brain}.lo${lo}.${spa_bw}.as${app_bw_spa}.ai${app_bw_int}.o


			# tester
			#hi_cmd="./../bin/cvcrf ${spa_bw} ${app_bw_spa} ${app_bw_int} ${brain} ${hi} "
			#lo_cmd="./../bin/cvcrf ${spa_bw} ${app_bw_spa} ${app_bw_int} ${brain} ${lo} "
			#echo ${lo_cmd}
			#echo ${hi_cmd}

			# call func
			export OMP_NUM_THREADS=$omp
			echo ${oname_lo}
			./../bin/cvcrf ${spa_bw} ${app_bw_spa} ${app_bw_int} ${brain} ${lo} > ${oname_lo}
			echo ${oname_hi}
			./../bin/cvcrf ${spa_bw} ${app_bw_spa} ${app_bw_int} ${brain} ${hi} > ${oname_hi}


		done # s loop
	done # k loop
done # m loop

