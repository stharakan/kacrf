#!/bin/bash

# Runs with one omp thread
#export OMP_NUM_THREADS=1
#echo "processing w/ 1 thread ."
#./../bin/tester 2500 > oscal_n50_t1.o
#echo "."
#./../bin/tester 10000 > oscal_n100_t1.o
#echo "."
#./../bin/tester 40000 > oscal_n200_t1.o
#echo "."
#./../bin/tester 160000 > oscal_n400_t1.o
#echo -e "\n"
#
#export OMP_NUM_THREADS=5
#echo "processing w/ 5 thread ."
#./../bin/tester 2500 > oscal_n50_t5.o
#echo "."
#./../bin/tester 10000 > oscal_n100_t5.o
#echo "."
#./../bin/tester 40000 > oscal_n200_t5.o
#echo "."
#./../bin/tester 160000 > oscal_n400_t5.o
#echo -e "\n"

export OMP_NUM_THREADS=10
echo "processing w/ 10 thread ."
./../bin/tester 2500 > oscal_n50_t10.o
echo "."
./../bin/tester 10000 > oscal_n100_t10.o
echo "."
./../bin/tester 40000 > oscal_n200_t10.o
echo "."
./../bin/tester 160000 > oscal_n400_t10.o
echo -e "\n"

export OMP_NUM_THREADS=20
echo "processing w/ 20 thread ."
./../bin/tester 2500 > oscal_n50_t20.o
echo "."
./../bin/tester 10000 > oscal_n100_t20.o
echo "."
./../bin/tester 40000 > oscal_n200_t20.o
echo "."
./../bin/tester 160000 > oscal_n400_t20.o
echo -e "\n"

#export OMP_NUM_THREADS=40
#./../bin/tester 2500 > oscal_n50_t10.o
#./../bin/tester 1000 > oscal_n100_t10.o
#./../bin/tester 4000 > oscal_n200_t10.o
#./../bin/tester 16000 > oscal_n400_t10.o
#./../bin/tester 64000 > oscal_n800_t10.o
