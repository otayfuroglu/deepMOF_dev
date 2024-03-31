#! /usr/bin/bash

numerOfFiles=$1
dir=$2
destinationDir=$3
for file in $(ls -p $dir | grep -v / | tail -$numerOfFiles);
do 
	mkdir -p $destinationDir
	mv $dir/$file $destinationDir/; 
done 
