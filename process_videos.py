"""
Compute image codes for every frame in a pair of videos.

Typical usage:

>>> python3 process_videos.py vidoes/tomato1.mov videos/tomato2.mov
"""

import os
import numpy as np
from argparse import ArgumentParser

from video_capture import iter_video
from image_encoding import ImageEncoder


if __name__ == "__main__":

    parser = ArgumentParser()
    parser.add_argument("filepaths", nargs="+")
    args = parser.parse_args()

    rootdir = os.path.dirname(__file__)
    codesdir = os.path.join(rootdir, "codes")
    os.makedirs(codesdir, exist_ok=True)

    filepaths = []    
    for fp in args.filepaths:
        if not os.path.isabs(fp):
            fp = os.path.join(rootdir, fp)
        filepaths.append(fp)
        if not os.path.isfile(fp):
            raise FileNotFoundError("File %r does not exist" % fp)

    encoder = ImageEncoder()

    for inpath in filepaths:
        firstname, _ = os.path.basename(inpath).split(".")
        outpath = os.path.join(codesdir, firstname + ".npy")
        print("Computing codes for %r . . ." % inpath)
        codes = np.array([encoder.encode(f) for f in iter_video(inpath)])
        print("Saving computed codes to %r . . ." % outpath)
        np.save(outpath, codes)
        print("Done processing %r.\n" % inpath)
