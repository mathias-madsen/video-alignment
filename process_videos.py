import os
import numpy as np
from argparse import ArgumentParser

from video_capture import iter_video
from image_encoding import ImageEncoder


if __name__ == "__main__":

    parser = ArgumentParser()
    parser.add_argument("name")
    args = parser.parse_args()
    name = args.name

    rootdir = os.path.dirname(__file__)

    vidpath1 = os.path.join(rootdir, "videos", "%s1.mov" % name)
    vidpath2 = os.path.join(rootdir, "videos", "%s2.mov" % name)

    codespath1 = os.path.join(rootdir, "codes", "%s1.npy" % name)
    codespath2 = os.path.join(rootdir, "codes", "%s2.npy" % name)

    encoder = ImageEncoder()
    codes1 = np.array([encoder.encode(f) for f in iter_video(vidpath1)])
    codes2 = np.array([encoder.encode(f) for f in iter_video(vidpath2)])

    os.makedirs(os.path.join(rootdir, "codes"), exist_ok=True)
    np.save(codespath1, codes1)
    np.save(codespath2, codes2)
