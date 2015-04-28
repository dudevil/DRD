# trimmer.cpp
simple utility for image cropping and resizing

Compile with cpp11

Usage:
./trimmer <input file> <output file> <output width>

Output image have size (output width)x(output width)

# run_trimmer.sh

Bash script for DRD datasets preprocessing.
For best experience, please LOOK at the script and analyze how it works. Put script near the "train" or "test" directory and run it as follows:

./run_trimmer.sh <train|test> <output width> <batch size>

first parameter: train or test or "folder name" - directory with images. Default value is "train"
script will take all *.jpeg images, results will be stored in PNG format.

second parameter: width of output image, integer (for example 256, 512, 1024). Default value is 512.

third parameter: batch size, amount of images in parallel batch. Less size -> more threads running parallel. Use with caution. Default value is 6000.



