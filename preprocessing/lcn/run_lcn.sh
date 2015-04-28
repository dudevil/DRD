#!/bin/bash

# default is train
set_name=${1:-train}

# gaussian kernel size, default is 17
kernel_size=${2:-17}

# If you want to convert image to grayscale
#gray=${3:0}

# Calculate LCN images
# You already SHOULD HAVE file lists to this moment

cd ${set_name}

for lst in ../${set_name}/_list_*;
do
    echo $lst;
    (while read line; do echo $line; ../lcn ${line%jpeg}png ${line%.jpeg}_lcn.png ${kernel_size} 0; done < $lst ) &
done;
wait

cd ..
