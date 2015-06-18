#!/bin/bash

# default is train
set_name=${1:-train}

# gaussian kernel size, default is 17
kernel_size=${2:-17}

# If you want to convert image to grayscale
gray=${3:-false}
gcn=${4:-false}
yuv=${5:-false}

name_gray=$([ ${gray} == "true" ] && echo "_gray" || echo "" )
name_gcn=$([ ${gcn} == "true" ] && echo "_gcn"  || echo "" )
name_yuv=$([ ${yuv} == "true" ] && echo "_yuv"  || echo "" )

# Calculate LCN images
# You already SHOULD HAVE file lists to this moment

cd ${set_name}

for lst in ../${set_name}_list_*;
do
    echo $lst;
    (while read line; do echo $line; outname=${line%.jpeg}_lcn_${kernel_size}${name_gray}${name_gcn}${name_yuv}.png; ../lcn -in=${line%jpeg}png -out=${outname} -ksize=${kernel_size} -gray=${gray} -yuv=${yuv} -gcn=${gcn}; done < $lst ) &
done;
wait

cd ..
