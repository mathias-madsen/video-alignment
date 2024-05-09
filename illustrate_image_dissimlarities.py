"""
Show pairs of best-match frames for a pair of videos, for diagnostics.

Usage like process_videos.py.
"""

import os
import numpy as np
from matplotlib import pyplot as plt
from argparse import ArgumentParser

from video_capture import iter_video
from shortest_path import compute_cost_to_go, compute_shortest_path


if __name__ == "__main__":

    parser = ArgumentParser()
    parser.add_argument("name")
    args = parser.parse_args()
    name = args.name

    rootdir = os.path.dirname(__file__)

    # load precomputed image encodings:
    codes1 = np.load(os.path.join(rootdir, "codes", "%s1.npy" % name))
    codes2 = np.load(os.path.join(rootdir, "codes", "%s2.npy" % name))

    # # downsample for the sake of efficiency:
    codes1 = codes1[:, ::5, :]
    codes2 = codes2[:, ::5, :]

    # flatten codes if they aren't already flat:
    vecs1 = np.reshape(codes1, [len(codes1), -1])
    vecs2 = np.reshape(codes2, [len(codes2), -1])

    # compute table of differences:
    norms = np.linalg.norm(vecs1[:, None] - vecs2[None, :], axis=-1)

    # load the videos:
    vidpath1 = os.path.join(rootdir, "videos", "%s1.mov" % name)
    vidpath2 = os.path.join(rootdir, "videos", "%s2.mov" % name)
    vid1 = np.array(list(iter_video(vidpath1)))
    vid2 = np.array(list(iter_video(vidpath2)))

    # plot some best-match example pairs:
    figure, axbox = plt.subplots(ncols=4, nrows=6, figsize=(14, 8))
    for row in axbox:
        for axes in row:
            idx1 = np.random.randint(len(vid1))
            idx2 = np.argmin(norms[idx1])
            dist = norms[idx1, idx2]
            combined = np.concatenate([vid1[idx1,], vid2[idx2,]], axis=1)
            axes.imshow(combined)
            axes.set_title("distance: %.2f" % dist)
    for ax in axbox.flatten():
        ax.axis("off")
    plt.tight_layout()
    plt.show()
