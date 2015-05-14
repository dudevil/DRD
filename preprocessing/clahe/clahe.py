__author__ = 'dudevil'

import os
import argparse
import time
from skimage import io
from skimage.exposure import equalize_adapthist
from multiprocessing import Pool


DATA_DIR = "data"

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-o",
                        "--output",
                        type=str,
                        default=os.path.join(DATA_DIR, "train", "clahe"),
                        help="Directory to put processed files to")
    parser.add_argument("-i",
                        "--input",
                        type=str,
                        default=os.path.join(DATA_DIR, "train", "resized"),
                        help="Directory to read files from")
    parser.add_argument("-j",
                        "--n_jobs",
                        type=int,
                        default=None,
                        help="Number of processes to start")

    args = parser.parse_args()
    n_jobs = args.n_jobs
    out_dir = args.output
    in_dir = args.input

    if not os.path.isdir(in_dir):
        print("Input directory %s does not exist!" % in_dir)
        exit()

    if not os.path.isdir(out_dir):
        print("Output directory %s does not exist!" % out_dir)
        exit()

    def perform_clage(image):
        path, fname = os.path.split(image)
        img = io.imread(image)
        img = equalize_adapthist(img)
        io.imsave(os.path.join(out_dir, fname), img)

    pool = Pool(n_jobs)
    infiles = map(lambda x: os.path.join(in_dir, x), os.listdir(in_dir))
    infiles = filter(lambda img: img.endswith('.png'), infiles)
    print("Processing files from %s to %s." % (in_dir, out_dir))
    start = time.time()
    pool.map(perform_clage, infiles)
    print("Processed %d images in %d seconds." % (len(infiles), time.time() - start))




