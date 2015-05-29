#!/usr/bin/env bash

# resize

echo "Step 1. Resize"

mogrify -format png -bordercolor black -border 1x1 -fuzz 10% -trim +repage -gravity center -resize 128 *_right.jpeg &

mogrify -format png -bordercolor black -border 1x1 -fuzz 10% -trim +repage -gravity center -resize 128 *_left.jpeg &

wait

#extent, two threads to speed-up process

echo "Step 2. Extent"

mogrify -format png -background black -gravity center -extent 128x128 *_left.png &

mogrify -format png -background black -gravity center -extent 128x128 *_right.png &

