#!/bin/bash
# Set defaults


# Process command line



# Make sbatch runfile



# If test, leave, o.w. sbatch and rm



usage="$(basename "$0") [-h] [-?] [-f ] -- bring file from s2:kacrf/ofiles to current dir

where:
	-h 	Show this help text
	-? 	Show this help text
	-f	specify the file (or just the last argument)"

while getopts "h?:f:" opt; do
	case "$opt" in
		h|\?)
			echo "$usage"
			exit 0
			;;
		f) FILE=$OPTARG
			;;
	esac
done
shift $((OPTIND-1))

# Check if a file is specified by option, otherwise it should be the last arg
if [ -z "$FILE" ]
then
	if [[ -a "$@" ]]
	then
		FILE="$@"
	else
		echo "No file specified"
		exit 0
	fi
fi

if [ ! $TEST ]
then
	scp tharakan@stampede2.tacc.utexas.edu:/home1/03158/tharakan/research/kacrf/ofiles/${FILE} ./
else
	echo "transfering \"$FILE\" to this dir"
fi
