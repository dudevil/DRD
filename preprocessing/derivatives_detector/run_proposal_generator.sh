#!/bin/bash

# default is train
set_name=${1:-train}

# default is 1024
threshold=${2:-300}

# images per thread (default is 6000)
batch_size=${3:-6000}

dark_only=${4:-1}
name_dark_only=$([ ${dark_only} == "0" ] && echo "dark_light" || echo "_dark_only" )

# list of files in directory
cd ${set_name}
ls *.png > ../${set_name}_list.txt

# Split list into parts and process them in parallel
split -d --additional-suffix=.txt -l $batch_size ../${set_name}_list.txt ../${set_name}_list_

# Trim images and resize to threshold
for lst in ../${set_name}_list_*;
do
    echo $lst;
    (while read line; do echo $line; ../proposal_generator $line ${line%.png}_blobs_${name_dark_only}.jpg ${threshold} ${dark_only}; done < $lst ) &
done;
wait

wait

cd ..
