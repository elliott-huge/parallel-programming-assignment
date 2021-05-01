#!/bin/sh

PLATFORM=$1
DEVICE=$2
RUNS=$3
VECTOR_LENGTH=$4
FILENAME=$5

echo "Running on platform $PLATFORM, device $DEVICE"
echo "Vector length: $VECTOR_LENGTH"

cd ~/OpenCL-Tutorials/x64/Debug/Tutorial\ 1

rm -f $FILENAME

for i in $(eval echo "{1..$RUNS}")
do
	echo "Run $i"
	./Tutorial\ 1.exe -p $PLATFORM -d $DEVICE -n $VECTOR_LENGTH | grep Total | awk '{ print $8 }' >> $5
done

echo "Output written to $FILENAME"