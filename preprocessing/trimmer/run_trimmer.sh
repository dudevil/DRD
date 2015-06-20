#!/bin/bash

# default is train
set_name=${1:-train}

# default is 1024
output_size=${2:-512}

# images per thread (default is 6000)
batch_size=${3:-6000}

inner=${4:-true}
name_inner=$([ ${inner} == "true" ] && echo "_inner" || echo "" )

# list of files in directory
cd ${set_name}
ls *.jpeg > ../${set_name}_list.txt

# Split list into parts and process them in parallel
split -d --additional-suffix=.txt -l $batch_size ../${set_name}_list.txt ../${set_name}_list_

# Trim images and resize to output_size
for lst in ../${set_name}_list_*;
do
    echo $lst;
    (while read line; do echo $line; ../trimmer $line ${line%.jpeg}${name_inner}.png ${output_size} ${inner}; done < $lst ) &
done;
wait

#echo "Step 2. Extent"
#extent, two threads to speed-up process
#mogrify -verbose -format png -background black -gravity center -extent ${output_size}x${output_size} *_left.png &
#mogrify -verbose -format png -background black -gravity center -extent ${output_size}x${output_size} *_right.png &
#wait

cd ..
